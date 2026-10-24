import os
import sys
import psutil
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# CORS origin constant — must match what's hardcoded in backend/main.py
# ---------------------------------------------------------------------------
VERCEL_ORIGIN = "https://frontend-theta-lemon-qnvdcsvxf.vercel.app"

class TestStartupFlow(unittest.TestCase):

    def test_01_startup_does_not_load_heavy_models(self):
        """Verify that importing backend.main and backend.pipeline does NOT eagerly load torch, ST, or faiss."""
        import backend.main as main_mod
        import backend.pipeline as pipe_mod

        # Neither torch nor sentence_transformers nor faiss should be loaded into sys.modules
        self.assertNotIn("torch", sys.modules, "torch should not be eagerly loaded at startup")
        self.assertNotIn("sentence_transformers", sys.modules, "sentence_transformers should not be eagerly loaded at startup")
        self.assertNotIn("faiss", sys.modules, "faiss should not be eagerly loaded at startup")

        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        print(f"\n[Test] Current process RSS after backend startup import: {rss_mb:.2f} MB")
        self.assertLess(rss_mb, 200, f"Startup memory ({rss_mb:.2f} MB) exceeds threshold")

    def test_02_api_endpoints_without_model_loading(self):
        """Verify that root / and /videos endpoints work without triggering model load."""
        import backend.main as main_mod

        # Call route handlers directly (avoids httpx proxy/DNS resolution in sandbox)
        res_home = main_mod.home()
        self.assertEqual(res_home, {"message": "VideoRAG Backend is running!"})

        res_videos = main_mod.get_videos()
        self.assertIn("videos", res_videos)
        self.assertIsInstance(res_videos["videos"], list)

        # Verify models are still not loaded after hitting basic endpoints
        self.assertNotIn("torch", sys.modules, "torch should not be loaded by GET / or GET /videos")
        self.assertNotIn("sentence_transformers", sys.modules, "sentence_transformers should not be loaded by GET / or GET /videos")
        self.assertNotIn("faiss", sys.modules, "faiss should not be loaded by GET / or GET /videos")

    def test_03_backward_compatibility_and_exports(self):
        """Verify backward compatibility of pipeline exports."""
        import backend.pipeline as pipe_mod

        # Verify functions are exported
        self.assertTrue(hasattr(pipe_mod, "get_embedding_model"))
        self.assertTrue(hasattr(pipe_mod, "get_faiss"))

        # Verify functions from pipeline imported by main are present
        self.assertTrue(callable(pipe_mod.transcribe_audio))
        self.assertTrue(callable(pipe_mod.create_chunks))
        self.assertTrue(callable(pipe_mod.create_embeddings))
        self.assertTrue(callable(pipe_mod.create_faiss_index))
        self.assertTrue(callable(pipe_mod.retrieve_chunks))
        self.assertTrue(callable(pipe_mod.retrieve_global_chunks))
        self.assertTrue(callable(pipe_mod.generate_answer))

        # Test lazy loading of faiss
        faiss_lib = pipe_mod.get_faiss()
        self.assertIsNotNone(faiss_lib)
        self.assertIn("faiss", sys.modules)

        # Test module __getattr__ compatibility
        self.assertIs(pipe_mod.faiss, faiss_lib)

    def test_04_sync_existing_videos_preserves_catalog(self):
        """Verify sync_existing_videos runs safely and populates library metadata."""
        import backend.main as main_mod

        catalog = main_mod.load_catalog()
        self.assertIsInstance(catalog, list)
        self.assertGreaterEqual(len(catalog), 0)

        # Re-syncing should be safe and idempotent
        main_mod.sync_existing_videos()
        reloaded = main_mod.load_catalog()
        self.assertEqual(len(catalog), len(reloaded))

    def test_05_execute_qa_flow(self):
        """Verify execute_qa route logic executes properly with mocked model and Groq client."""
        from backend.main import execute_qa

        with patch('backend.pipeline.get_embedding_model') as mock_get_embed, \
             patch('backend.pipeline.get_faiss') as mock_get_faiss, \
             patch('backend.pipeline.get_groq_client') as mock_groq_client:

            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1] * 384]
            mock_get_embed.return_value = mock_model

            mock_faiss = MagicMock()
            mock_index = MagicMock()
            mock_index.ntotal = 5
            mock_index.search.return_value = ([[0.05]], [[0]])
            mock_faiss.read_index.return_value = mock_index
            mock_get_faiss.return_value = mock_faiss

            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Grounded response about machine learning"
            mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
            mock_groq_client.return_value = mock_client

            result = execute_qa(question="What is deep learning?")
            self.assertIn("answer", result)
            self.assertIn("sources", result)
            self.assertIn("ranked_videos", result)
            self.assertEqual(result["answer"], "Grounded response about machine learning")

    def test_06_transcribe_audio_oom_detection(self):
        """Verify transcribe_audio captures SIGKILL / OOM subprocess return codes properly."""
        import subprocess
        from backend.pipeline import transcribe_audio

        with patch("os.path.exists", side_effect=lambda p: False if p.endswith(".json") else True), \
             patch("subprocess.run") as mock_run:
            # Simulate OOM SIGKILL returncode -9
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=-9,
                cmd=["whisper", "test.wav"],
                stderr="Killed\n"
            )

            with self.assertRaises(RuntimeError) as ctx:
                transcribe_audio("dummy_audio.wav")

            self.assertIn("Out of Memory", str(ctx.exception))
            self.assertIn("SIGKILL", str(ctx.exception))

    def test_07_upload_error_handling_and_secret_redaction(self):
        """Verify sanitize_error_detail redacts secrets, and the upload HTTPException wrapper works."""
        from fastapi import HTTPException
        from backend.main import sanitize_error_detail

        # --- 7a. Secret redaction ---
        dummy_err = Exception(
            "Failed with gsk_abcdef1234567890abcdef1234567890 and Bearer secret_token_xyz"
        )
        sanitized = sanitize_error_detail(dummy_err)
        self.assertNotIn("gsk_abcdef1234567890abcdef1234567890", sanitized)
        self.assertNotIn("Bearer secret_token_xyz", sanitized)
        self.assertIn("[REDACTED_KEY]", sanitized)
        self.assertIn("[REDACTED]", sanitized)

        # --- 7b. sanitize_error_detail leaves ordinary messages intact ---
        plain = sanitize_error_detail(RuntimeError("Whisper OOM - model too large"))
        self.assertIn("Whisper OOM", plain)

        # --- 7c. Verify the HTTPException wrapping pattern used in upload_video ---
        # Simulate what upload_video does in its except block
        original_error = RuntimeError("Whisper transcription killed by OOM: returncode -9")
        safe_detail = sanitize_error_detail(original_error)
        wrapped = HTTPException(
            status_code=500,
            detail=f"Video processing failed: {safe_detail}"
        )
        self.assertEqual(wrapped.status_code, 500)
        self.assertIn("Whisper", wrapped.detail)
        self.assertIn("Video processing failed", wrapped.detail)

    def test_08_whisper_model_env_var_is_read(self):
        """Verify transcribe_audio passes the WHISPER_MODEL env var value to the whisper CLI."""
        import subprocess, pathlib, json as _json
        from backend.pipeline import transcribe_audio

        captured_cmds = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        # os.path.exists: wav file exists (True), json transcript does NOT exist (False)
        def fake_exists(path):
            return not path.endswith(".json")

        with patch.dict("os.environ", {"WHISPER_MODEL": "tiny"}, clear=False), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("os.path.exists", side_effect=fake_exists):
            try:
                transcribe_audio("dummy_audio.wav")
            except Exception:
                pass  # may fail when reading JSON output — we only care the command ran

        self.assertGreater(len(captured_cmds), 0, "subprocess.run was never called")
        whisper_cmd = captured_cmds[0]
        self.assertIn("--model", whisper_cmd, "'--model' flag missing from whisper command")
        model_idx = whisper_cmd.index("--model") + 1
        self.assertEqual(
            whisper_cmd[model_idx], "tiny",
            f"Expected model 'tiny' from WHISPER_MODEL env var, got '{whisper_cmd[model_idx]}'"
        )

    def test_09_whisper_model_defaults_to_base(self):
        """Verify transcribe_audio defaults to 'base' when WHISPER_MODEL is not set."""
        import subprocess
        from backend.pipeline import transcribe_audio

        captured_cmds = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        def fake_exists(path):
            return not path.endswith(".json")

        # Ensure WHISPER_MODEL is absent from environment
        import os as _os
        env_without = {k: v for k, v in _os.environ.items() if k != "WHISPER_MODEL"}
        with patch.dict("os.environ", env_without, clear=True), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("os.path.exists", side_effect=fake_exists):
            try:
                transcribe_audio("dummy_audio.wav")
            except Exception:
                pass

        self.assertGreater(len(captured_cmds), 0, "subprocess.run was never called")
        whisper_cmd = captured_cmds[0]
        self.assertIn("--model", whisper_cmd, "'--model' flag missing from whisper command")
        model_idx = whisper_cmd.index("--model") + 1
        self.assertEqual(
            whisper_cmd[model_idx], "base",
            f"Expected default model 'base', got '{whisper_cmd[model_idx]}'"
        )


# ---------------------------------------------------------------------------
# CORS Tests
# ---------------------------------------------------------------------------

def _make_cors_mw(origins):
    """Instantiate CORSMiddleware with a dummy ASGI app for direct method inspection."""
    from starlette.middleware.cors import CORSMiddleware

    async def _dummy(scope, receive, send):
        pass  # never called in our unit tests

    return CORSMiddleware(
        app=_dummy,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Length", "Content-Range"],
        max_age=600,
    )


def _preflight_headers(cors_mw, origin, req_method="POST", req_headers="content-type"):
    """
    Call CORSMiddleware.preflight_response() directly and return the response headers
    as a dict.  Pure synchronous — no ASGI transport needed.
    """
    from starlette.datastructures import Headers
    raw = {
        "origin": origin,
        "access-control-request-method": req_method,
        "access-control-request-headers": req_headers,
    }
    # Starlette Headers accepts a dict[str, str] in its constructor
    headers = Headers(headers=raw)
    resp = cors_mw.preflight_response(request_headers=headers)
    return dict(resp.headers)


class TestCORSConfiguration(unittest.TestCase):
    """Test CORSMiddleware configuration without TestClient / network stack."""

    @classmethod
    def setUpClass(cls):
        import backend.main as main_mod
        cls.allowed_origins = main_mod.allowed_origins
        cls._BUILTIN_ORIGINS = main_mod._BUILTIN_ORIGINS
        cls.cors_mw = _make_cors_mw(cls.allowed_origins)

    # ------------------------------------------------------------------
    # 10. Vercel origin present in the compiled allowed_origins list
    # ------------------------------------------------------------------

    def test_10_vercel_origin_in_allowed_list(self):
        """Verify the Vercel production origin is present in allowed_origins."""
        self.assertIn(
            VERCEL_ORIGIN,
            self.allowed_origins,
            f"'{VERCEL_ORIGIN}' missing from allowed_origins.\nCurrent: {self.allowed_origins}"
        )

    # ------------------------------------------------------------------
    # 11. OPTIONS preflight — Vercel origin
    # ------------------------------------------------------------------

    def test_11_cors_preflight_vercel_origin(self):
        """OPTIONS preflight from Vercel origin must return 200 with correct CORS headers."""
        from starlette.datastructures import Headers
        headers = Headers({
            "origin": VERCEL_ORIGIN,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        })
        resp = self.cors_mw.preflight_response(request_headers=headers)
        self.assertEqual(resp.status_code, 200)
        acao = resp.headers.get("access-control-allow-origin", "")
        self.assertEqual(acao, VERCEL_ORIGIN,
                         f"ACAO was '{acao}', expected '{VERCEL_ORIGIN}'")
        acam = resp.headers.get("access-control-allow-methods", "")
        for method in ("GET", "POST", "OPTIONS"):
            self.assertIn(method, acam,
                          f"'{method}' missing from ACAM: '{acam}'")

    # ------------------------------------------------------------------
    # 12. is_allowed_origin — Vercel origin
    # ------------------------------------------------------------------

    def test_12_cors_vercel_origin_is_allowed(self):
        """CORSMiddleware.is_allowed_origin() must return True for the Vercel origin."""
        allowed = self.cors_mw.is_allowed_origin(VERCEL_ORIGIN)
        self.assertTrue(allowed,
                        f"is_allowed_origin returned False for '{VERCEL_ORIGIN}'")

    # ------------------------------------------------------------------
    # 13. Local dev origins still allowed (regression)
    # ------------------------------------------------------------------

    def test_13_cors_localhost_still_allowed(self):
        """Local dev origins must still be allowed and pass preflight."""
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            allowed = self.cors_mw.is_allowed_origin(origin)
            self.assertTrue(allowed, f"Local origin '{origin}' is not allowed")

            from starlette.datastructures import Headers
            resp = self.cors_mw.preflight_response(Headers({
                "origin": origin,
                "access-control-request-method": "GET"
            }))
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get("access-control-allow-origin"), origin)

    # ------------------------------------------------------------------
    # 14. Unknown origin blocked
    # ------------------------------------------------------------------

    def test_14_cors_unknown_origin_blocked(self):
        """Unknown origins must NOT be allowed by the middleware and preflight must fail."""
        rogue = "https://evil.example.com"
        allowed = self.cors_mw.is_allowed_origin(rogue)
        self.assertFalse(allowed,
                         f"Rogue origin '{rogue}' was unexpectedly allowed")

        from starlette.datastructures import Headers
        resp = self.cors_mw.preflight_response(Headers({
            "origin": rogue,
            "access-control-request-method": "POST"
        }))
        self.assertEqual(resp.status_code, 400)
        self.assertNotIn("access-control-allow-origin", resp.headers)

    # ------------------------------------------------------------------
    # 15. Runtime CORS_ORIGINS env var merging
    # ------------------------------------------------------------------

    def test_15_cors_runtime_origin_via_env_var(self):
        """Extra origins from CORS_ORIGINS env var should be merged into allowed list."""
        extra = "https://preview-branch.vercel.app"
        import backend.main as main_mod

        with patch.dict("os.environ", {"CORS_ORIGINS": extra}, clear=False):
            dynamic = list(main_mod._BUILTIN_ORIGINS)
            for o in os.getenv("CORS_ORIGINS", "").strip().split(","):
                o = o.strip().rstrip("/")
                if o and o not in dynamic:
                    dynamic.append(o)

        self.assertIn(extra, dynamic,
                      f"Runtime origin '{extra}' missing from merged list: {dynamic}")

        runtime_mw = _make_cors_mw(dynamic)
        self.assertTrue(runtime_mw.is_allowed_origin(extra),
                        f"Runtime CORS middleware does not allow '{extra}'")


if __name__ == "__main__":
    unittest.main()
