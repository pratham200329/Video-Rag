import os
import sys
import psutil
import unittest
from unittest.mock import patch, MagicMock

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

if __name__ == "__main__":
    unittest.main()
