import { useState, useRef, useEffect } from "react";

// Normalize API base URL from Vite environment variable with local fallback
const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL;
const API_URL = (RAW_API_BASE && typeof RAW_API_BASE === "string" && RAW_API_BASE.trim())
  ? RAW_API_BASE.trim().replace(/\/+$/, "")
  : "http://127.0.0.1:8000";

// Helper component for markdown rendering without external dependencies
function FormattedMessage({ text }) {
  if (!text) return null;

  // Split into paragraphs / lines
  const lines = text.split("\n");
  const elements = [];
  let currentList = null;

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    if (!trimmed) {
      if (currentList) {
        elements.push(currentList);
        currentList = null;
      }
      return;
    }

    // Bold title / heading (e.g. ### Heading or **Title**)
    if (trimmed.startsWith("###") || trimmed.startsWith("##")) {
      if (currentList) {
        elements.push(currentList);
        currentList = null;
      }
      const titleText = trimmed.replace(/^#+\s*/, "");
      elements.push(
        <h4 key={`h-${index}`} className="msg-heading">
          {titleText}
        </h4>
      );
      return;
    }

    // Bullet item (e.g. * item or - item)
    if (trimmed.startsWith("* ") || trimmed.startsWith("- ") || trimmed.startsWith("• ")) {
      const itemContent = trimmed.substring(2);
      if (!currentList || currentList.type !== "ul") {
        if (currentList) elements.push(currentList);
        currentList = {
          type: "ul",
          key: `ul-${index}`,
          items: []
        };
      }
      currentList.items.push(
        <li key={`li-${index}`}>
          {formatInline(itemContent)}
        </li>
      );
      return;
    }

    // Numbered item (e.g. 1. item)
    const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      const itemContent = numMatch[2];
      if (!currentList || currentList.type !== "ol") {
        if (currentList) elements.push(currentList);
        currentList = {
          type: "ol",
          key: `ol-${index}`,
          items: []
        };
      }
      currentList.items.push(
        <li key={`oli-${index}`}>
          {formatInline(itemContent)}
        </li>
      );
      return;
    }

    // Standard paragraph line
    if (currentList) {
      elements.push(currentList);
      currentList = null;
    }

    elements.push(
      <p key={`p-${index}`} className="msg-paragraph">
        {formatInline(trimmed)}
      </p>
    );
  });

  if (currentList) {
    elements.push(currentList);
  }

  return (
    <div className="formatted-markdown-body">
      {elements.map((el) => {
        if (el.type === "ul") {
          return (
            <ul key={el.key} className="msg-bullet-list">
              {el.items}
            </ul>
          );
        }
        if (el.type === "ol") {
          return (
            <ol key={el.key} className="msg-ordered-list">
              {el.items}
            </ol>
          );
        }
        return el;
      })}
    </div>
  );
}

// Inline formatting for bold (**bold**) and inline code (`code`)
function formatInline(text) {
  const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i} className="msg-inline-code">{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function App() {
  // TOP NAVIGATION TAB: "home" (default) | "videos" | "chat"
  const [activeTab, setActiveTab] = useState("home");

  // UPLOAD / LINK INGESTION MODAL
  const [showUpload, setShowUpload] = useState(false);
  const [uploadTab, setUploadTab] = useState("file"); // "file" | "link"
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  // VIDEO LIBRARY & ACTIVE PLAYER
  const [library, setLibrary] = useState([]);
  const [loadingLibrary, setLoadingLibrary] = useState(false);
  const [videoData, setVideoData] = useState(null);
  const [pendingTimestamp, setPendingTimestamp] = useState(null);
  const [activeSourceIndex, setActiveSourceIndex] = useState(null);

  // STUDIO QUICK QA (WITHIN VIDEOS TAB)
  const [studioQuestion, setStudioQuestion] = useState("");
  const [studioAnswer, setStudioAnswer] = useState("");
  const [studioSources, setStudioSources] = useState([]);
  const [studioRankedVideos, setStudioRankedVideos] = useState([]);
  const [studioAsking, setStudioAsking] = useState(false);
  const [studioScope, setStudioScope] = useState("current"); // "current" | "all"

  // DEDICATED AI CHAT SECTION STATE
  const [chatMode, setChatMode] = useState("all"); // "all" | "selected"
  const [selectedChatVideoId, setSelectedChatVideoId] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatAsking, setChatAsking] = useState(false);
  const [chatCopiedIndex, setChatCopiedIndex] = useState(null);

  // NOTIFICATION & TOAST STATE
  const [toastMessage, setToastMessage] = useState("");
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  // VIDEO REFERENCE
  const videoRef = useRef(null);
  const chatBottomRef = useRef(null);

  // Show transient toast
  const showToast = (msg, duration = 3000) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage((prev) => (prev === msg ? "" : prev));
    }, duration);
  };

  // =====================================================
  // FETCH VIDEO LIBRARY
  // =====================================================
  const fetchLibrary = async () => {
    try {
      setLoadingLibrary(true);
      const res = await fetch(`${API_URL}/videos`);
      if (res.ok) {
        const data = await res.json();
        const vids = data.videos || [];
        setLibrary(vids);

        // Auto-select first ready video if none is selected
        setVideoData((current) => {
          if (!current && vids.length > 0) {
            const readyVid = vids.find((v) => v.status === "ready") || vids[0];
            return readyVid;
          }
          return current;
        });

        // Initialize selectedChatVideoId if empty
        setSelectedChatVideoId((currentId) => {
          if (!currentId && vids.length > 0) {
            const firstReady = vids.find((v) => v.status === "ready") || vids[0];
            return firstReady.filename || firstReady.id;
          }
          return currentId;
        });

        return vids;
      }
    } catch (err) {
      console.warn("Could not fetch video library:", err);
    } finally {
      setLoadingLibrary(false);
    }
    return [];
  };

  useEffect(() => {
    fetchLibrary();
  }, []);

  // Scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (activeTab === "chat" && chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, chatAsking, activeTab]);

  // =====================================================
  // SELECT VIDEO FOR STUDIO PLAYBACK
  // =====================================================
  const selectVideoForPlayback = (video) => {
    setVideoData(video);
    setPendingTimestamp(null);
    setActiveTab("videos");
    showToast(`Loaded ${video.filename} in Studio Player`);
  };

  // =====================================================
  // CITATION SEEK & AUTOPLAY (WORKS ACROSS TABS)
  // =====================================================
  const jumpToSource = (source, index = null) => {
    if (index !== null) {
      setActiveSourceIndex(index);
    }

    const targetFilename = source.filename;
    setActiveTab("videos");

    // Check if target is same as current video
    if (!targetFilename || (videoData && videoData.filename === targetFilename)) {
      if (videoRef.current) {
        videoRef.current.currentTime = source.start;
        videoRef.current.play().catch((err) => {
          console.log("Video play prevented:", err);
        });
      }
      showToast(`Playing ${targetFilename || "video"} at ${formatTime(source.start)}`);
    } else {
      // Citation points to a DIFFERENT video: switch video and seek on metadata load
      const foundVideo = library.find((v) => v.filename === targetFilename) || {
        filename: targetFilename,
        id: source.video_id
      };
      setPendingTimestamp(source.start);
      setVideoData(foundVideo);
      showToast(`Switching to ${targetFilename} at ${formatTime(source.start)}`);
    }
  };

  // Format seconds to MM:SS
  const formatTime = (seconds) => {
    if (seconds === undefined || seconds === null || isNaN(seconds)) return "00:00";
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  };

  // =====================================================
  // SEND MESSAGE IN DEDICATED AI CHAT
  // =====================================================
  const sendChatMessage = async (presetText = null) => {
    const query = (presetText || chatInput).trim();
    if (!query || chatAsking) return;

    const userMessage = {
      id: "u_" + Date.now(),
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };

    setMessages((prev) => [...prev, userMessage]);
    setChatInput("");
    setChatAsking(true);
    setError("");

    // Prepare payload
    const targetVideoId = chatMode === "all" ? "all" : (selectedChatVideoId || "all");

    // Map conversation history (last 8 turns)
    const historyPayload = messages.slice(-8).map((m) => ({
      role: m.role,
      content: m.content
    }));

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          video_id: targetVideoId,
          history: historyPayload
        })
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        id: "a_" + Date.now(),
        role: "assistant",
        content: data.answer || "No response received.",
        sources: data.sources || [],
        ranked_videos: data.ranked_videos || [],
        modeUsed: chatMode,
        targetVideoName: chatMode === "selected" ? selectedChatVideoId : null,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      };

      setMessages((prev) => [...prev, assistantMessage]);

    } catch (err) {
      console.error("Chat error:", err);
      const errorMessage = {
        id: "err_" + Date.now(),
        role: "assistant",
        content: `Error: Unable to retrieve answer. ${err.message || "Please check backend connection."}`,
        isError: true,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setChatAsking(false);
    }
  };

  // Clear chat conversation
  const clearChatHistory = () => {
    setMessages([]);
    showToast("Conversation cleared");
  };

  // =====================================================
  // STUDIO QUICK QA (INSIDE VIDEOS TAB)
  // =====================================================
  const askStudioQuestion = async () => {
    if (!studioQuestion.trim() || studioAsking) return;

    setStudioAsking(true);
    setError("");

    try {
      let targetVid = "all";
      if (studioScope === "current" && videoData?.filename) {
        targetVid = videoData.filename;
      }

      const url = `${API_URL}/ask?question=${encodeURIComponent(studioQuestion.trim())}&video_id=${encodeURIComponent(targetVid)}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error("Question processing failed");
      }

      const data = await response.json();
      setStudioAnswer(data.answer);
      setStudioSources(data.sources || []);
      setStudioRankedVideos(data.ranked_videos || []);
      setActiveSourceIndex(null);

    } catch (err) {
      console.error(err);
      setError("Unable to obtain studio answer. Check backend connection.");
    } finally {
      setStudioAsking(false);
    }
  };

  // =====================================================
  // UPLOAD LOCAL VIDEO FILE
  // =====================================================
  const uploadVideo = async () => {
    if (!file || uploading) return;

    setUploading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      setVideoData(data);
      setShowUpload(false);
      setFile(null);
      await fetchLibrary();
      showToast(`Uploaded and indexed ${data.filename}!`);

    } catch (err) {
      console.error(err);
      setError("Video processing failed. Please verify the backend connection.");
    } finally {
      setUploading(false);
    }
  };

  // =====================================================
  // PROCESS REMOTE VIDEO LINK (YOUTUBE / DIRECT)
  // =====================================================
  const processVideoLink = async () => {
    if (!videoUrl.trim() || uploading) return;

    setUploading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/process-link`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoUrl.trim() })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to process video link.");
      }

      const data = await response.json();
      setVideoData(data);
      setShowUpload(false);
      setVideoUrl("");
      await fetchLibrary();
      showToast(`Downloaded and indexed ${data.filename}!`);

    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to download and process video link.");
    } finally {
      setUploading(false);
    }
  };

  // Preset quick prompt chips
  const quickPrompts = [
    "Summarize the key takeaways and conclusions",
    "What are the core technical arguments presented?",
    "Explain the main problem being addressed",
    "Compare the different concepts discussed across videos"
  ];

  // Helper to copy text to clipboard
  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setChatCopiedIndex(index);
    setTimeout(() => setChatCopiedIndex(null), 2000);
  };

  // Compute library stats
  const readyVideosCount = library.filter((v) => v.status === "ready").length;
  const currentChatVideo = library.find((v) => v.filename === selectedChatVideoId || v.id === selectedChatVideoId);
  const isSelectedChatVideoReady = currentChatVideo?.status === "ready";

  return (
    <div className="app-container">
      {/* AMBIENT BACKGROUND GLOWS */}
      <div className="ambient-sphere sphere-cyan" />
      <div className="ambient-sphere sphere-mint" />
      <div className="ambient-sphere sphere-cobalt" />
      <div className="noise-texture-overlay" />
      <div className="grid-blueprint-overlay" />

      {/* =====================================================
          UNIVERSAL TOP NAVIGATION BAR
          ===================================================== */}
      <header className="navbar-container">
        <nav className="navbar-glass">
          {/* Brand Logo - Returns to Home */}
          <div className="brand-group" onClick={() => setActiveTab("home")} style={{ cursor: "pointer" }} title="Return to Home">
            <div className="brand-logo-mark">
              <svg className="brand-svg" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="url(#brandGrad)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="url(#brandGrad)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="url(#brandGrad)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                <defs>
                  <linearGradient id="brandGrad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#00F2FE" />
                    <stop offset="0.5" stopColor="#00E5FF" />
                    <stop offset="1" stopColor="#10B981" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div className="brand-text-wrap">
              <span className="brand-name">VideoRAG</span>
              <span className="brand-tag">PRO</span>
            </div>
          </div>

          {/* PRIMARY NAVIGATION TABS: [ HOME ], [ VIDEOS ], [ AI CHAT ] */}
          <div className="primary-nav-tabs">
            <button
              className={`nav-tab-pill ${activeTab === "home" ? "active" : ""}`}
              onClick={() => setActiveTab("home")}
              aria-label="Navigate to Home"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tab-icon">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                <polyline points="9 22 9 12 15 12 15 22" />
              </svg>
              <span>Home</span>
            </button>

            <button
              className={`nav-tab-pill ${activeTab === "videos" ? "active" : ""}`}
              onClick={() => setActiveTab("videos")}
              aria-label="Navigate to Videos"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tab-icon">
                <polygon points="23 7 16 12 23 17 23 7" />
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
              </svg>
              <span>Videos</span>
              <span className="nav-count-badge">{library.length}</span>
            </button>

            <button
              className={`nav-tab-pill ${activeTab === "chat" ? "active" : ""}`}
              onClick={() => setActiveTab("chat")}
              aria-label="Navigate to AI Chat"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tab-icon">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              <span>AI Chat</span>
              <span className="nav-status-dot-active" />
            </button>
          </div>

          {/* Status Chip & Actions */}
          <div className="nav-actions-group">
            <div className="radar-status-chip" title={`${readyVideosCount} of ${library.length} videos indexed in FAISS`}>
              <span className="radar-ping" />
              <span className="radar-dot" />
              <span className="radar-label">{readyVideosCount} Videos Ready</span>
            </div>

            <button
              className="nav-cta-btn"
              onClick={() => setShowUpload(true)}
              title="Upload file or ingest video link"
            >
              <span>+ Upload Video</span>
            </button>
          </div>
        </nav>
      </header>

      {/* =====================================================
          TAB 0: DEDICATED HOME / LANDING PAGE
          ===================================================== */}
      {activeTab === "home" && (
        <main className="tab-content-container home-page-container">
          {/* HERO SECTION */}
          <section className="home-hero-section">
            <div className="home-hero-badge">
              <span className="badge-spark">✦</span>
              <span>Intelligent Neural Video Intelligence</span>
            </div>

            <h1 className="home-hero-headline">
              Your Videos. Your Knowledge.<br />
              <span className="home-hero-gradient-text">One AI.</span>
            </h1>

            <p className="home-hero-description">
              Ask natural language questions across all your video content and receive grounded,
              hallucination-free answers with precise, clickable timestamp citations that play on demand.
            </p>

            <div className="home-hero-actions">
              <button
                className="home-primary-cta-btn"
                onClick={() => setActiveTab("videos")}
              >
                <span>Start Exploring</span>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="cta-arrow">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>

              <button
                className="home-secondary-cta-btn"
                onClick={() => setShowUpload(true)}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <span>Upload a Video</span>
              </button>
            </div>
          </section>

          {/* COMPACT FEATURE CARDS (3 CARDS) */}
          <section className="home-features-section">
            <div className="home-feature-card">
              <div className="feature-icon-box cyan">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <h3>Ask Anything</h3>
              <p>
                Query complex lectures, meetings, and tutorials in plain English.
                Get clear, concise answers synthesized strictly from transcribed speech.
              </p>
            </div>

            <div className="home-feature-card">
              <div className="feature-icon-box mint">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="2" y1="12" x2="22" y2="12" />
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                </svg>
              </div>
              <h3>Search Across Videos</h3>
              <p>
                Find relevant topics across your entire processed library simultaneously,
                or narrow your questions to a single selected video.
              </p>
            </div>

            <div className="home-feature-card">
              <div className="feature-icon-box emerald">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <polyline points="12 6 12 12 16 14" />
                </svg>
              </div>
              <h3>Timestamp Citations</h3>
              <p>
                Every answer comes with verifiable timestamp cards. Click any citation
                to switch videos and seek to the exact second in the player.
              </p>
            </div>
          </section>

          {/* LIVE LIBRARY OVERVIEW BANNER */}
          <section className="home-library-overview-card">
            <div className="overview-left">
              <div className="overview-title-row">
                <span className="overview-pill-tag">Live Knowledge Vault</span>
                <span className="overview-count-pill">{library.length} Registered Videos</span>
                <span className="overview-ready-pill">{readyVideosCount} Ready for RAG</span>
              </div>
              <p className="overview-subtitle">
                {readyVideosCount > 0
                  ? `${readyVideosCount} videos are indexed and ready for immediate natural language Q&A.`
                  : "Upload videos or paste links to build your searchable neural video library."}
              </p>

              {/* Quick preview pills */}
              <div className="overview-video-pills">
                {library.slice(0, 4).map((vid) => (
                  <span
                    key={vid.id || vid.filename}
                    className="video-pill-preview"
                    onClick={() => selectVideoForPlayback(vid)}
                    title={`Click to open ${vid.filename} in Studio`}
                  >
                    📹 {vid.filename}
                    <span className={`mini-status ${vid.status === "ready" ? "ready" : "uploaded"}`}>
                      {vid.status === "ready" ? "Ready" : "Uploaded"}
                    </span>
                  </span>
                ))}
              </div>
            </div>

            <div className="overview-right">
              <button
                className="overview-explore-btn"
                onClick={() => setActiveTab("videos")}
              >
                <span>View Full Video Vault ›</span>
              </button>
            </div>
          </section>

          {/* HOW IT WORKS SECTION */}
          <section className="home-how-section">
            <div className="how-header">
              <span className="how-step-eyebrow">End-to-End Pipeline</span>
              <h2>How VideoRAG Works</h2>
              <p>From raw video files to grounded, timestamped intelligence in seconds.</p>
            </div>

            <div className="how-steps-grid">
              <div className="how-step-card">
                <div className="how-step-badge">01</div>
                <h4>Upload</h4>
                <p>Drop an MP4, MOV, or WEBM video, or paste a link from YouTube.</p>
              </div>

              <div className="how-step-card">
                <div className="how-step-badge">02</div>
                <h4>Process</h4>
                <p>FFmpeg extracts 16kHz audio, Whisper transcribes timestamps, and FAISS vectors are indexed.</p>
              </div>

              <div className="how-step-card">
                <div className="how-step-badge">03</div>
                <h4>Ask</h4>
                <p>Ask in natural language across all videos or focus specifically on one video.</p>
              </div>

              <div className="how-step-card">
                <div className="how-step-badge">04</div>
                <h4>Discover</h4>
                <p>Receive synthesized answers with clickable timestamp citations that autoplay instantly.</p>
              </div>
            </div>
          </section>

          {/* PRODUCT FOOTER */}
          <footer className="home-footer">
            <div className="footer-content">
              <div className="footer-brand">
                <span className="brand-name">VideoRAG AI</span>
                <span className="footer-tagline">Conversational Neural Video Intelligence Platform</span>
              </div>
              <div className="footer-meta">
                <span>Obsidian Noir & Electric Cyan Edition</span>
                <span>•</span>
                <span>Powered by Whisper & Groq LLaMA</span>
              </div>
            </div>
          </footer>
        </main>
      )}

      {/* =====================================================
          TAB 1: VIDEOS SECTION (STUDIO PLAYER + FULL LIBRARY)
          ===================================================== */}
      {activeTab === "videos" && (
        <main className="tab-content-container">
          {/* Studio Bar */}
          <div className="studio-control-banner">
            <div className="studio-banner-left">
              <span className="studio-tag">Active Studio Video</span>
              {library.length > 0 ? (
                <div className="video-selector-wrapper">
                  <select
                    className="workspace-video-select"
                    value={videoData?.filename || ""}
                    onChange={(e) => {
                      const chosen = library.find((v) => v.filename === e.target.value);
                      if (chosen) selectVideoForPlayback(chosen);
                    }}
                    title="Switch active video in player"
                  >
                    {library.map((v) => (
                      <option key={v.id || v.filename} value={v.filename}>
                        📹 {v.filename} {v.status === "ready" ? "(Ready)" : `(${v.status})`}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <span className="active-video-label">{videoData?.filename || "No video selected"}</span>
              )}
            </div>

            <div className="studio-banner-right">
              <button
                className="switch-to-chat-btn"
                onClick={() => {
                  if (videoData?.filename) {
                    setSelectedChatVideoId(videoData.filename);
                    setChatMode("selected");
                  }
                  setActiveTab("chat");
                }}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <span>Ask about this video in AI Chat ›</span>
              </button>
            </div>
          </div>

          {/* TWO-COLUMN STUDIO GRID */}
          <div className="workspace-stage-grid">
            {/* LEFT COLUMN: THEATER PLAYER & VIDEO META */}
            <section className="theater-column">
              <div className="theater-player-bezel">
                <div className="video-ambilight-glow" />
                <div className="video-viewport-frame">
                  {videoData?.filename ? (
                    <video
                      ref={videoRef}
                      controls
                      className="html5-video-player"
                      src={`${API_URL}/uploads/${encodeURIComponent(videoData.filename)}`}
                      onLoadedMetadata={() => {
                        if (pendingTimestamp !== null && videoRef.current) {
                          videoRef.current.currentTime = pendingTimestamp;
                          videoRef.current.play().catch((err) => {
                            console.log("Autoplay interrupted:", err);
                          });
                          setPendingTimestamp(null);
                        }
                      }}
                    />
                  ) : (
                    <div className="player-empty-placeholder" onClick={() => setShowUpload(true)}>
                      <div className="empty-play-icon">▶</div>
                      <p>Select any video from the library below or upload a new one to begin.</p>
                      <button className="upload-inline-btn">+ Upload Video</button>
                    </div>
                  )}
                </div>
              </div>

              {/* VIDEO METADATA CARD */}
              {videoData && (
                <div className="video-details-card">
                  <div className="details-header-row">
                    <div className="title-block">
                      <h2 className="media-headline">{videoData.filename}</h2>
                      <p className="media-subtext">
                        {videoData.status === "ready"
                          ? "Dense FAISS embeddings and Whisper speech alignment active."
                          : `Status: ${videoData.status || "processing"} — awaiting vector indexing.`}
                      </p>
                    </div>

                    <div className="engine-tags-row">
                      <div className="spec-tag">
                        <span className={`spec-indicator ${videoData.status === "ready" ? "mint" : "amber"}`} />
                        <span>{videoData.status === "ready" ? "RAG Ready" : "Unindexed"}</span>
                      </div>
                      {videoData.file_size && (
                        <div className="spec-tag">
                          <span>{(videoData.file_size / (1024 * 1024)).toFixed(1)} MB</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </section>

            {/* RIGHT COLUMN: STUDIO QUICK-QA PANEL */}
            <section className="copilot-column">
              <div className="copilot-terminal-card">
                <div className="copilot-header-bar">
                  <div className="copilot-identity-wrap">
                    <div className="copilot-avatar-orb">
                      <span className="avatar-core-pulse" />
                    </div>
                    <div>
                      <h3 className="copilot-name">Studio Quick QA</h3>
                      <span className="copilot-grounding-text">Direct In-Player Timestamp Seeking</span>
                    </div>
                  </div>

                  <div className="scope-toggle-group">
                    <button
                      className={`scope-pill-btn ${studioScope === "current" ? "active" : ""}`}
                      onClick={() => setStudioScope("current")}
                      title="Search only currently playing video"
                    >
                      Active Video
                    </button>
                    <button
                      className={`scope-pill-btn ${studioScope === "all" ? "active" : ""}`}
                      onClick={() => setStudioScope("all")}
                      title="Search all indexed videos"
                    >
                      All Videos
                    </button>
                  </div>
                </div>

                <div className="chat-stream-viewport studio-stream">
                  {!studioAnswer && !studioAsking && (
                    <div className="copilot-empty-state">
                      <div className="empty-spark-icon">✦</div>
                      <h4>Ask about this video</h4>
                      <p>Type any question below. Citations will appear with exact timestamps that automatically seek the player.</p>
                      <div className="suggested-prompts-stack">
                        {quickPrompts.slice(0, 3).map((prompt, idx) => (
                          <button
                            key={idx}
                            className="prompt-pill-chip"
                            onClick={() => {
                              setStudioQuestion(prompt);
                            }}
                          >
                            <span>✦</span>
                            <span>{prompt}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {studioAsking && (
                    <div className="copilot-thinking-card">
                      <div className="audio-wave-bars">
                        <span className="wave-bar bar-1" />
                        <span className="wave-bar bar-2" />
                        <span className="wave-bar bar-3" />
                        <span className="wave-bar bar-4" />
                        <span className="wave-bar bar-5" />
                      </div>
                      <div className="thinking-copy-col">
                        <strong>Retrieving timestamps & synthesizing answer...</strong>
                        <span>Grounded strictly on speech transcripts</span>
                      </div>
                    </div>
                  )}

                  {studioAnswer && !studioAsking && (
                    <div className="synthesized-response-card">
                      <div className="response-top-row">
                        <span className="response-source-tag">✦ Grounded Answer</span>
                        <button
                          className="action-copy-btn"
                          onClick={() => copyToClipboard(studioAnswer, "studio")}
                        >
                          {chatCopiedIndex === "studio" ? "Copied!" : "Copy"}
                        </button>
                      </div>

                      <div className="response-text-body">
                        <FormattedMessage text={studioAnswer} />
                      </div>

                      {studioSources.length > 0 && (
                        <div className="citations-tray">
                          <div className="citations-tray-header">
                            <span className="tray-title">Verified Video Citations</span>
                            <span className="tray-hint">Click citation to seek player</span>
                          </div>

                          <div className="citations-cards-stack">
                            {studioSources.map((source, index) => (
                              <button
                                key={index}
                                className={`interactive-citation-card ${activeSourceIndex === index ? "active-playing" : ""}`}
                                onClick={() => jumpToSource(source, index)}
                              >
                                <div className="citation-play-disc">▶</div>
                                <div className="citation-info-col">
                                  <div className="citation-meta-row">
                                    <span className="citation-video-badge">{source.filename || "Video"}</span>
                                    <span className="citation-badge-text">
                                      {formatTime(source.start)} — {formatTime(source.end)}
                                    </span>
                                  </div>
                                  {source.text && <p className="citation-excerpt">{source.text}</p>}
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Studio Input */}
                <div className="command-bar-footer">
                  <div className="command-bar-capsule">
                    <input
                      type="text"
                      className="command-bar-input"
                      placeholder={`Ask about ${studioScope === "current" ? (videoData?.filename || "video") : "all videos"}...`}
                      value={studioQuestion}
                      onChange={(e) => setStudioQuestion(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") askStudioQuestion();
                      }}
                    />
                    <button
                      className="execute-query-btn"
                      onClick={askStudioQuestion}
                      disabled={!studioQuestion.trim() || studioAsking}
                    >
                      {studioAsking ? <span className="btn-mini-spinner" /> : "Ask"}
                    </button>
                  </div>
                </div>
              </div>
            </section>
          </div>

          {/* =====================================================
              PERSISTENT VIDEO LIBRARY GALLERY (ALWAYS ACCESSIBLE)
              ===================================================== */}
          <section className="library-section" id="library">
            <div className="library-section-header">
              <div>
                <div className="library-title-row">
                  <h2 className="library-section-title">Video Knowledge Vault</h2>
                  <span className="library-count-pill">{library.length} Videos Total</span>
                  <span className="library-ready-pill">{readyVideosCount} Ready for RAG</span>
                </div>
                <p className="library-section-subtitle">
                  Select any video to open directly in the Studio Player or query in multi-turn AI Chat.
                </p>
              </div>

              <button className="shelf-add-btn" onClick={() => setShowUpload(true)}>
                <span>+ Upload New Video</span>
              </button>
            </div>

            <div className="library-cards-grid">
              {library.map((vid) => {
                const isActive = videoData?.filename === vid.filename;
                const isReady = vid.status === "ready";
                return (
                  <div
                    key={vid.id || vid.filename}
                    className={`library-video-card ${isActive ? "active-card" : ""}`}
                  >
                    <div className="library-card-top">
                      <div className="library-card-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polygon points="23 7 16 12 23 17 23 7" />
                          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                        </svg>
                      </div>
                      <div className="library-card-meta">
                        <span className="library-card-title" title={vid.filename}>
                          {vid.filename}
                        </span>
                        <span className="library-card-info">
                          {vid.file_size ? `${(vid.file_size / (1024 * 1024)).toFixed(1)} MB` : "Video Knowledge"}
                          {isActive && " · Currently Loaded"}
                        </span>
                      </div>
                    </div>

                    <div className="library-card-bottom">
                      {isReady ? (
                        <span className="status-pill-ready">
                          <span className="mini-pulse-dot" />
                          Ready for RAG
                        </span>
                      ) : (
                        <span className="status-pill-processing">
                          <span>●</span>
                          {vid.status || "Processing"}
                        </span>
                      )}

                      <div className="card-actions-group">
                        <button
                          className="card-action-btn primary"
                          onClick={() => selectVideoForPlayback(vid)}
                          title="Open in player"
                        >
                          <span>Open in Studio</span>
                          <span>›</span>
                        </button>
                        <button
                          className="card-action-btn secondary"
                          onClick={() => {
                            setSelectedChatVideoId(vid.filename);
                            setChatMode("selected");
                            setActiveTab("chat");
                          }}
                          title="Open scoped chat"
                        >
                          <span>Ask in Chat</span>
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Upload Card */}
              <div className="library-add-card" onClick={() => setShowUpload(true)}>
                <div className="add-card-circle">+</div>
                <span className="add-card-text">Upload Video Knowledge Base</span>
              </div>
            </div>
          </section>

          {/* THREE CORE ARCHITECTURAL PILLARS */}
          <div className="pillars-grid" id="capabilities">
            <div className="pillar-card">
              <div className="pillar-icon-box">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              </div>
              <span className="pillar-step-tag">STAGE 01</span>
              <h3>Whisper Speech AI</h3>
              <p>FFmpeg extracts 16,000 Hz mono audio. OpenAI Whisper generates word-level timestamps and segment transcriptions.</p>
            </div>

            <div className="pillar-card spotlight">
              <div className="pillar-icon-box">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                  <polyline points="2 17 12 22 22 17"/>
                  <polyline points="2 12 12 17 22 12"/>
                </svg>
              </div>
              <span className="pillar-step-tag">STAGE 02</span>
              <h3>FAISS Vector Indexing</h3>
              <p>Dense semantic representations with SentenceTransformer. Sub-second nearest-neighbor similarity retrieval across hours of video.</p>
            </div>

            <div className="pillar-card">
              <div className="pillar-icon-box">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <polygon points="10 8 16 12 10 16 10 8"/>
                </svg>
              </div>
              <span className="pillar-step-tag">STAGE 03</span>
              <h3>Precision Citations</h3>
              <p>LLaMA synthesizes grounded answers with verifiable timestamp cards. Click any citation to seek and autoplay instantly.</p>
            </div>
          </div>
        </main>
      )}

      {/* =====================================================
          TAB 2: DEDICATED AI CHAT SECTION (FULL-FEATURED)
          ===================================================== */}
      {activeTab === "chat" && (
        <main className="tab-content-container chat-tab-layout">
          <div className="chat-workspace-card">
            {/* CHAT CONTROL HEADER */}
            <div className="chat-top-control-bar">
              <div className="chat-control-left">
                <div className="chat-avatar-badge">
                  <span className="avatar-core-pulse" />
                </div>
                <div>
                  <h2 className="chat-view-title">VideoRAG AI Assistant</h2>
                  <span className="chat-view-subtitle">
                    Multi-turn conversational reasoning · Zero hallucination
                  </span>
                </div>
              </div>

              {/* MODE SWITCHER: ALL VIDEOS vs SELECTED VIDEO */}
              <div className="chat-mode-selector-wrap">
                <div className="mode-pill-toggle">
                  <button
                    className={`mode-pill-btn ${chatMode === "all" ? "active" : ""}`}
                    onClick={() => setChatMode("all")}
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="2" y1="12" x2="22" y2="12" />
                      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                    </svg>
                    <span>Search All Videos</span>
                  </button>

                  <button
                    className={`mode-pill-btn ${chatMode === "selected" ? "active" : ""}`}
                    onClick={() => setChatMode("selected")}
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <polygon points="10 8 16 12 10 16 10 8" />
                    </svg>
                    <span>Search Selected Video</span>
                  </button>
                </div>

                {/* VIDEO SELECTOR DROPDOWN (ONLY VISIBLE IN SELECTED MODE) */}
                {chatMode === "selected" && (
                  <div className="selected-video-dropdown-box">
                    <select
                      className="chat-video-select"
                      value={selectedChatVideoId}
                      onChange={(e) => setSelectedChatVideoId(e.target.value)}
                      title="Select video to focus your questions on"
                    >
                      {library.map((vid) => (
                        <option key={vid.id || vid.filename} value={vid.filename}>
                          📹 {vid.filename} {vid.status === "ready" ? "(Ready)" : `(${vid.status})`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {/* RIGHT CONTROLS: CLEAR & RETURN TO STUDIO */}
              <div className="chat-control-right">
                {messages.length > 0 && (
                  <button className="chat-action-text-btn" onClick={clearChatHistory} title="Reset chat history">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                    <span>Clear Chat</span>
                  </button>
                )}

                <button
                  className="chat-action-text-btn highlight"
                  onClick={() => setActiveTab("videos")}
                  title="Switch to studio player"
                >
                  <span>Go to Studio Player ›</span>
                </button>
              </div>
            </div>

            {/* UNPROCESSED VIDEO WARNING BANNER */}
            {chatMode === "selected" && currentChatVideo && !isSelectedChatVideoReady && (
              <div className="chat-status-notice-banner warning">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <span>
                  <strong>Notice:</strong> &apos;{currentChatVideo.filename}&apos; has status &apos;{currentChatVideo.status}&apos; and is not yet indexed with FAISS. Queries will report its status rather than searching unrelated videos.
                </span>
              </div>
            )}

            {/* CONVERSATION STREAM VIEWPORT */}
            <div className="chat-messages-scroll-area">
              {/* WELCOME / EMPTY STATE */}
              {messages.length === 0 && (
                <div className="chat-hero-empty-state">
                  <div className="empty-spark-orb">✦</div>
                  <h3>Conversational Video Intelligence</h3>
                  <p>
                    {chatMode === "all"
                      ? "Global Mode Active: Asking questions searches across all indexed videos simultaneously. Results are ranked by similarity with exact timestamp citations."
                      : `Scoped Mode Active: Questions are focused strictly on '${selectedChatVideoId || "selected video"}'.`}
                  </p>

                  <div className="chat-prompt-suggestions-grid">
                    <span className="suggestions-label">Try starting with:</span>
                    <div className="suggestions-pills-row">
                      {quickPrompts.map((promptText, idx) => (
                        <button
                          key={idx}
                          className="chat-quick-prompt-pill"
                          onClick={() => sendChatMessage(promptText)}
                        >
                          <span className="spark">✦</span>
                          <span>{promptText}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* MESSAGE THREAD */}
              {messages.map((msg, index) => {
                const isUser = msg.role === "user";
                return (
                  <div
                    key={msg.id || index}
                    className={`chat-message-row ${isUser ? "user-row" : "assistant-row"}`}
                  >
                    {!isUser && (
                      <div className="assistant-avatar-badge">
                        <span>AI</span>
                      </div>
                    )}

                    <div className={`chat-message-bubble ${isUser ? "user-bubble" : "assistant-bubble"} ${msg.isError ? "error-bubble" : ""}`}>
                      <div className="bubble-header-line">
                        <span className="bubble-author-name">{isUser ? "You" : "VideoRAG AI"}</span>
                        {msg.modeUsed && !isUser && (
                          <span className="bubble-mode-tag">
                            {msg.modeUsed === "all" ? "🌐 Global Search" : `🎯 Scoped: ${msg.targetVideoName || "Single Video"}`}
                          </span>
                        )}
                        <span className="bubble-time-stamp">{msg.timestamp}</span>
                      </div>

                      <div className="bubble-content-text">
                        <FormattedMessage text={msg.content} />
                      </div>

                      {/* RANKED VIDEOS BADGES (ASSISTANT MESSAGES) */}
                      {msg.ranked_videos && msg.ranked_videos.length > 0 && (
                        <div className="ranked-videos-tray">
                          <div className="ranked-tray-header">
                            <span className="ranked-tray-title">
                              <span>✦</span> Matched Video Rankings
                            </span>
                            <span className="ranked-tray-subtitle">Ranked by semantic distance</span>
                          </div>
                          <div className="ranked-pills-row">
                            {msg.ranked_videos.map((rv, rIdx) => (
                              <div
                                key={rv.filename || rIdx}
                                className={`ranked-video-pill ${rIdx === 0 ? "top-rank" : ""}`}
                              >
                                <span className="rank-badge-tag">
                                  {rIdx === 0 ? "🥇 #1" : rIdx === 1 ? "🥈 #2" : `🥉 #${rIdx + 1}`}
                                </span>
                                <span className="rank-video-name">{rv.filename}</span>
                                <span className="rank-dist-pill">L2: {rv.min_distance.toFixed(2)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* VERIFIED TIMESTAMP CITATIONS (CLICK TO SWITCH & SEEK) */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="citations-tray">
                          <div className="citations-tray-header">
                            <span className="tray-title">Verified Video Citations ({msg.sources.length})</span>
                            <span className="tray-hint">Click citation to jump directly to video & timestamp</span>
                          </div>

                          <div className="citations-cards-stack">
                            {msg.sources.map((source, sIdx) => (
                              <button
                                key={sIdx}
                                className="interactive-citation-card"
                                onClick={() => jumpToSource(source)}
                                title={`Jump to ${source.filename || "video"} at ${formatTime(source.start)}`}
                              >
                                <div className="citation-play-disc">▶</div>
                                <div className="citation-info-col">
                                  <div className="citation-meta-row">
                                    <span className="citation-video-badge">{source.filename || "Video"}</span>
                                    <span className="citation-badge-text">
                                      {formatTime(source.start)} — {formatTime(source.end)}
                                    </span>
                                  </div>
                                  {source.text && <p className="citation-excerpt">{source.text}</p>}
                                </div>
                                <div className="citation-action-icon">›</div>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* COPY BUTTON */}
                      {!isUser && !msg.isError && (
                        <div className="bubble-footer-actions">
                          <button
                            className="bubble-copy-btn"
                            onClick={() => copyToClipboard(msg.content, index)}
                          >
                            {chatCopiedIndex === index ? "✓ Copied" : "Copy answer"}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* LOADING WAVEFORM CARD */}
              {chatAsking && (
                <div className="chat-message-row assistant-row">
                  <div className="assistant-avatar-badge">
                    <span>AI</span>
                  </div>
                  <div className="chat-message-bubble assistant-bubble thinking-bubble">
                    <div className="copilot-thinking-card">
                      <div className="audio-wave-bars">
                        <span className="wave-bar bar-1" />
                        <span className="wave-bar bar-2" />
                        <span className="wave-bar bar-3" />
                        <span className="wave-bar bar-4" />
                        <span className="wave-bar bar-5" />
                      </div>
                      <div className="thinking-copy-col">
                        <strong>
                          {chatMode === "all"
                            ? "Searching across all video knowledge vaults..."
                            : `Searching knowledge vault for '${selectedChatVideoId}'...`}
                        </strong>
                        <span>Grounding response strictly in transcript timestamps with Groq LLaMA</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatBottomRef} />
            </div>

            {/* CHAT INPUT COMMAND BAR */}
            <div className="chat-input-container">
              <div className="chat-input-capsule">
                <input
                  type="text"
                  className="chat-text-input"
                  placeholder={
                    chatMode === "all"
                      ? "Ask any question across your entire video library... (Enter to send)"
                      : `Ask a question specifically about '${selectedChatVideoId || "selected video"}'...`
                  }
                  value={chatInput}
                  disabled={chatAsking}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendChatMessage();
                    }
                  }}
                />

                <div className="chat-input-actions">
                  <button
                    className="chat-send-btn"
                    onClick={() => sendChatMessage()}
                    disabled={!chatInput.trim() || chatAsking}
                    aria-label="Send message"
                  >
                    {chatAsking ? (
                      <span className="btn-mini-spinner" />
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <line x1="22" y1="2" x2="11" y2="13"/>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                      </svg>
                    )}
                  </button>
                </div>
              </div>

              <div className="chat-footer-caption">
                <span>
                  ✦ Answers are synthesized strictly from transcribed speech. Click citations to jump straight to relevant video moments.
                </span>
              </div>
            </div>
          </div>
        </main>
      )}

      {/* =====================================================
          REDESIGNED UPLOAD & INGESTION MODAL (PHASE 2)
          ===================================================== */}
      {showUpload && (
        <div
          className="modal-overlay-backdrop"
          onClick={() => {
            if (!uploading) setShowUpload(false);
          }}
          role="dialog"
          aria-modal="true"
          aria-labelledby="upload-modal-title"
        >
          <div className="modal-window-card" onClick={(e) => e.stopPropagation()}>
            {/* Modal Exit Button */}
            <button
              className="modal-exit-btn"
              onClick={() => setShowUpload(false)}
              disabled={uploading}
              aria-label="Close dialog"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>

            {/* Modal Header Block */}
            <div className="modal-header-block">
              <div className={`modal-hero-icon ${uploading ? "loading-active" : ""}`}>
                {uploading ? (
                  <div className="loading-spinner-ring" />
                ) : (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="23 7 16 12 23 17 23 7" />
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                  </svg>
                )}
              </div>
              <h2 id="upload-modal-title">{uploading ? "Analyzing Video Pipeline" : "Ingest Video Knowledge"}</h2>
              <p className="modal-lead-text">
                {uploading
                  ? "Extracting audio, running Whisper speech recognition, and building FAISS vector index..."
                  : "Upload a local video file or paste a video link to index with AI."}
              </p>
            </div>

            {/* SEGMENTED TAB SWITCHER */}
            <div className="modal-tabs-pill-row">
              <button
                type="button"
                className={`modal-tab-btn ${uploadTab === "file" ? "active" : ""}`}
                disabled={uploading}
                onClick={() => setUploadTab("file")}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tab-btn-icon">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <span>Upload File</span>
              </button>

              <button
                type="button"
                className={`modal-tab-btn ${uploadTab === "link" ? "active" : ""}`}
                disabled={uploading}
                onClick={() => setUploadTab("link")}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tab-btn-icon">
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                </svg>
                <span>Paste Video Link</span>
              </button>
            </div>

            {/* TAB 1: FILE DROPZONE & SELECTION */}
            {uploadTab === "file" && (
              <div className="upload-file-container">
                <input
                  ref={fileInputRef}
                  type="file"
                  id="modal-video-input"
                  className="sr-only-file-input"
                  accept="video/mp4,video/quicktime,video/webm,video/x-matroska"
                  disabled={uploading}
                  onChange={(e) => {
                    if (e.target.files?.[0]) setFile(e.target.files[0]);
                  }}
                />

                {!file ? (
                  <div
                    className={`dropzone-surface ${isDragging ? "dragging-over" : ""}`}
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => {
                      e.preventDefault();
                      if (!uploading) setIsDragging(true);
                    }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setIsDragging(false);
                      if (uploading) return;
                      if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
                    }}
                  >
                    <div className="dropzone-interactive-content">
                      <div className="dropzone-icon-box">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                          <polyline points="17 8 12 3 7 8"/>
                          <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                      </div>
                      <div className="dropzone-text-group">
                        <span className="dropzone-primary-text">Drag your video here or browse files</span>
                        <span className="dropzone-format-tags">Supported: MP4, MOV, WEBM, MKV · Max recommended: 1 GB</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="file-selected-card">
                    <div className="file-selected-left">
                      <div className="file-selected-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polygon points="23 7 16 12 23 17 23 7" />
                          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                        </svg>
                      </div>
                      <div className="file-selected-meta">
                        <span className="file-name-bold" title={file.name}>{file.name}</span>
                        <div className="file-size-row">
                          <span className="file-size-tag">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                          <span className="file-ready-tag">✓ Ready to process</span>
                        </div>
                      </div>
                    </div>

                    <div className="file-selected-actions">
                      <button
                        type="button"
                        className="file-change-btn"
                        disabled={uploading}
                        onClick={() => fileInputRef.current?.click()}
                      >
                        Change
                      </button>
                      <button
                        type="button"
                        className="file-remove-btn"
                        disabled={uploading}
                        onClick={() => setFile(null)}
                        title="Remove file"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: LINK INGESTION */}
            {uploadTab === "link" && (
              <div className="link-input-surface">
                <div className="link-field-box">
                  <div className="link-field-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                    </svg>
                  </div>
                  <input
                    type="url"
                    className="link-input-control"
                    placeholder="Paste YouTube, Vimeo, or direct video link..."
                    value={videoUrl}
                    disabled={uploading}
                    onChange={(e) => setVideoUrl(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && videoUrl.trim()) processVideoLink();
                    }}
                  />
                  {videoUrl && (
                    <button
                      type="button"
                      className="clear-input-btn"
                      onClick={() => setVideoUrl("")}
                      disabled={uploading}
                      aria-label="Clear URL"
                    >
                      ×
                    </button>
                  )}
                </div>
                <div className="link-platforms-row">
                  <span className="platform-tag">YouTube</span>
                  <span className="platform-tag">Vimeo</span>
                  <span className="platform-tag">MP4 / WEBM</span>
                  <span className="platform-hint">Auto-downloaded & indexed</span>
                </div>
              </div>
            )}

            {/* PIPELINE PROGRESS TRACKER (DURING UPLOAD/PROCESSING) */}
            {uploading && (
              <div className="pipeline-progress-box">
                <div className="progress-track-wrapper">
                  <div className="progress-gradient-bar" />
                </div>
                <div className="progress-steps-list">
                  <div className="progress-step-item active">
                    <span className="step-bullet" />
                    <span>1. Ingestion / Download</span>
                  </div>
                  <div className="progress-step-item active">
                    <span className="step-bullet" />
                    <span>2. FFmpeg Audio Extraction</span>
                  </div>
                  <div className="progress-step-item active">
                    <span className="step-bullet" />
                    <span>3. Whisper Speech AI</span>
                  </div>
                  <div className="progress-step-item active">
                    <span className="step-bullet" />
                    <span>4. FAISS Vector Space</span>
                  </div>
                </div>
              </div>
            )}

            {/* ERROR DISPLAY */}
            {error && (
              <div className="modal-error-toast">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <span>{error}</span>
              </div>
            )}

            {/* SUBMIT BUTTON */}
            <button
              className="modal-submit-action-btn"
              disabled={
                uploading ||
                (uploadTab === "file" && !file) ||
                (uploadTab === "link" && !videoUrl.trim())
              }
              onClick={() => {
                if (uploadTab === "file") uploadVideo();
                else processVideoLink();
              }}
            >
              <span>
                {uploading
                  ? "Processing Video Knowledge..."
                  : uploadTab === "file"
                  ? "Upload & Process"
                  : "Download & Index"}
              </span>
            </button>
          </div>
        </div>
      )}

      {/* =====================================================
          FLOATING TOAST NOTIFICATION
          ===================================================== */}
      {toastMessage && (
        <div className="floating-action-toast">
          <span className="toast-spark">✦</span>
          <span>{toastMessage}</span>
        </div>
      )}
    </div>
  );
}

export default App;
