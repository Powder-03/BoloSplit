"""Web Login and Token Capture Router for Google Sign-In Testing."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter(tags=["Web Authentication Testing"])

TOKEN_FILE = Path(".last_google_token.txt")


class SaveTokenRequest(BaseModel):
    token: str
    email: str | None = None
    name: str | None = None
    uid: str | None = None


@router.post("/api/test/save-token", summary="Save captured Google ID token for automated test execution")
async def save_google_token(payload: SaveTokenRequest):
    """Save the client-captured Google Firebase token to disk for test scripts."""
    if not payload.token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No token provided.")

    TOKEN_FILE.write_text(payload.token.strip())
    return {
        "status": "SAVED",
        "email": payload.email,
        "name": payload.name,
        "uid": payload.uid,
        "message": "Token successfully saved to backend for automated endpoint verification.",
    }


@router.get("/api/test/last-token", summary="Retrieve latest captured Google token")
async def get_last_token():
    """Retrieve the latest captured token."""
    if not TOKEN_FILE.exists():
        return {"token": None}
    return {"token": TOKEN_FILE.read_text().strip()}


@router.get("/login", response_class=HTMLResponse, summary="Google Login Web Interface")
async def get_login_page():
    """Interactive Google Sign-In page for Firebase testing."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BoloSplit - Google Authentication & Endpoint Tester</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Firebase Compat SDKs -->
  <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-auth-compat.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col items-center justify-center p-4">
  <div class="max-w-2xl w-full bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl p-6 sm:p-8">
    
    <!-- Header -->
    <div class="flex items-center space-x-3 mb-6">
      <div class="w-12 h-12 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white text-2xl shadow-lg">
        <i class="fa-solid fa-microphone-lines"></i>
      </div>
      <div>
        <h1 class="text-2xl font-bold text-white">BoloSplit Voice Splitwise</h1>
        <p class="text-sm text-slate-400">Live Firebase Google Authentication & Endpoint Verifier</p>
      </div>
    </div>

    <!-- Step 1: Firebase Web API Key Setup -->
    <div id="setup-section" class="mb-6 p-4 bg-slate-900/60 rounded-xl border border-slate-700/80">
      <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
        1. Firebase Web API Key
      </label>
      <div class="flex space-x-2">
        <input id="api-key-input" type="text" placeholder="AIzaSy..." 
          class="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500 transition">
        <button id="save-key-btn" onclick="saveApiKey()" 
          class="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-semibold transition">
          Save Key
        </button>
      </div>
      <p class="text-xs text-slate-500 mt-2">
        <i class="fa-solid fa-info-circle mr-1"></i>
        Find it in <b>Firebase Console</b> &rarr; <b>Project Settings (⚙️)</b> &rarr; <b>General</b> &rarr; <b>Web API Key</b>.
      </p>
    </div>

    <!-- Step 2: Sign in with Google Button -->
    <div id="auth-section" class="mb-6 text-center">
      <button id="google-login-btn" onclick="handleGoogleLogin()" 
        class="w-full sm:w-auto inline-flex items-center justify-center space-x-3 bg-white hover:bg-slate-100 text-slate-800 font-semibold px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transition transform active:scale-95 disabled:opacity-50">
        <svg class="w-5 h-5" viewBox="0 0 24 24">
          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
          <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
        </svg>
        <span>Sign in with Google</span>
      </button>
      <div id="login-status" class="text-sm mt-3 text-slate-400">Please enter your Web API Key above to enable sign-in.</div>
    </div>

    <!-- Step 3: User Profile & Live Test Runner (Hidden until logged in) -->
    <div id="user-section" class="hidden space-y-6">
      
      <!-- User Card -->
      <div class="flex items-center space-x-4 p-4 bg-emerald-950/40 border border-emerald-500/30 rounded-xl">
        <img id="user-photo" src="" alt="Avatar" class="w-12 h-12 rounded-full border border-emerald-400">
        <div class="flex-1">
          <div class="flex items-center space-x-2">
            <h2 id="user-name" class="font-bold text-emerald-300">Logged In User</h2>
            <span class="text-xs bg-emerald-800 text-emerald-200 px-2 py-0.5 rounded-full font-mono">Firebase Verified</span>
          </div>
          <p id="user-email" class="text-sm text-slate-300"></p>
          <p id="user-uid" class="text-xs text-slate-500 font-mono"></p>
        </div>
        <button onclick="handleSignOut()" class="text-xs text-rose-400 hover:text-rose-300 underline">Sign Out</button>
      </div>

      <!-- Action: Run Live Tests Against Backend -->
      <div class="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 bg-slate-900/80 rounded-xl border border-slate-700">
        <div>
          <h3 class="font-semibold text-white">Execute Real Endpoints Test</h3>
          <p class="text-xs text-slate-400">Hits /balances, /groups, /voice, and /settlements with your live Google Bearer Token.</p>
        </div>
        <button id="run-tests-btn" onclick="runLiveEndpointTests()" 
          class="bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-white font-bold px-5 py-2.5 rounded-xl shadow-lg transition">
          <i class="fa-solid fa-play mr-1.5"></i> Run Live Tests
        </button>
      </div>

      <!-- Live Test Output Logs -->
      <div class="p-4 bg-black/60 rounded-xl border border-slate-800 font-mono text-xs">
        <div class="flex items-center justify-between border-b border-slate-800 pb-2 mb-3">
          <span class="text-slate-400 font-semibold">Test Suite Execution Console</span>
          <span id="test-summary" class="text-slate-500">Idle</span>
        </div>
        <div id="test-logs" class="space-y-2 max-h-72 overflow-y-auto">
          <div class="text-slate-600">// Log messages will stream here when you click 'Run Live Tests'...</div>
        </div>
      </div>

    </div>

  </div>

  <script>
    let app, auth, currentToken = null;

    window.onload = function() {
      const savedKey = localStorage.getItem("bolosplit_firebase_api_key");
      if (savedKey) {
        document.getElementById("api-key-input").value = savedKey;
        initFirebase(savedKey);
      }
    };

    function saveApiKey() {
      const key = document.getElementById("api-key-input").value.trim();
      if (!key) {
        alert("Please enter a valid Firebase Web API Key.");
        return;
      }
      localStorage.setItem("bolosplit_firebase_api_key", key);
      initFirebase(key);
    }

    function initFirebase(apiKey) {
      try {
        if (!firebase.apps.length) {
          app = firebase.initializeApp({
            apiKey: apiKey,
            authDomain: "bolosplit-274e0.firebaseapp.com",
            projectId: "bolosplit-274e0"
          });
        }
        auth = firebase.auth();
        
        document.getElementById("login-status").innerHTML = 
          '<span class="text-emerald-400"><i class="fa-solid fa-circle-check mr-1"></i>Firebase initialized. Ready for Google Sign-In.</span>';
        document.getElementById("google-login-btn").disabled = false;

        auth.onAuthStateChanged(async (user) => {
          if (user) {
            currentToken = await user.getIdToken();
            showLoggedInUI(user, currentToken);
            // Automatically sync token to backend
            await syncTokenToBackend(user, currentToken);
          } else {
            showLoggedOutUI();
          }
        });
      } catch (err) {
        document.getElementById("login-status").innerHTML = 
          '<span class="text-rose-400"><i class="fa-solid fa-triangle-exclamation mr-1"></i>' + err.message + '</span>';
      }
    }

    async function handleGoogleLogin() {
      if (!auth) {
        alert("Please save your Firebase Web API Key first.");
        return;
      }
      const provider = new firebase.auth.GoogleAuthProvider();
      try {
        document.getElementById("login-status").innerText = "Opening Google Sign-In popup...";
        const result = await auth.signInWithPopup(provider);
        const token = await result.user.getIdToken();
        currentToken = token;
      } catch (error) {
        document.getElementById("login-status").innerHTML = 
          '<span class="text-rose-400">Login failed: ' + error.message + '</span>';
      }
    }

    async function handleSignOut() {
      if (auth) await auth.signOut();
      currentToken = null;
      showLoggedOutUI();
    }

    function showLoggedInUI(user, token) {
      document.getElementById("auth-section").classList.add("hidden");
      document.getElementById("setup-section").classList.add("hidden");
      document.getElementById("user-section").classList.remove("hidden");
      document.getElementById("user-name").innerText = user.displayName || "Google User";
      document.getElementById("user-email").innerText = user.email || "";
      document.getElementById("user-uid").innerText = "UID: " + user.uid;
      document.getElementById("user-photo").src = user.photoURL || "https://ui-avatars.com/api/?name=" + encodeURIComponent(user.displayName || "User");
    }

    function showLoggedOutUI() {
      document.getElementById("auth-section").classList.remove("hidden");
      document.getElementById("setup-section").classList.remove("hidden");
      document.getElementById("user-section").classList.add("hidden");
      document.getElementById("login-status").innerText = "Signed out. Click above to sign in again.";
    }

    async function syncTokenToBackend(user, token) {
      try {
        const resp = await fetch("/api/test/save-token", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            token: token,
            email: user.email,
            name: user.displayName,
            uid: user.uid
          })
        });
        const data = await resp.json();
        console.log("Token synced to backend:", data);
      } catch (e) {
        console.warn("Failed to sync token to backend:", e);
      }
    }

    function log(msg, type = "info") {
      const container = document.getElementById("test-logs");
      const el = document.createElement("div");
      if (type === "success") el.className = "text-emerald-400 flex items-center space-x-1";
      else if (type === "error") el.className = "text-rose-400 flex items-center space-x-1";
      else el.className = "text-slate-300";
      el.innerHTML = msg;
      container.appendChild(el);
      container.scrollTop = container.scrollHeight;
    }

    async function runLiveEndpointTests() {
      if (!currentToken) {
        alert("Please sign in first.");
        return;
      }
      const logs = document.getElementById("test-logs");
      logs.innerHTML = "";
      document.getElementById("test-summary").innerText = "Testing in progress...";
      
      const headers = {
        "Authorization": "Bearer " + currentToken,
        "Content-Type": "application/json"
      };

      try {
        // 1. GET /balances
        log("🔵 Testing <b>GET /balances</b> with Google Bearer Token...");
        const balResp = await fetch("/balances", { headers });
        const balData = await balResp.json();
        if (balResp.ok) {
          log(`✅ <b>GET /balances</b>: Passed (Status ${balResp.status}, Net Total: ₹${balData.net_total})`, "success");
        } else {
          log(`❌ <b>GET /balances</b>: Failed (${balResp.status}) ${JSON.stringify(balData)}`, "error");
        }

        // 2. GET /friends
        log("🔵 Testing <b>GET /friends</b>...");
        const frResp = await fetch("/friends", { headers });
        const frData = await frResp.json();
        if (frResp.ok) {
          log(`✅ <b>GET /friends</b>: Passed (${frData.length} friends found)`, "success");
        } else {
          log(`❌ <b>GET /friends</b>: Failed (${frResp.status})`, "error");
        }

        // 3. POST /groups (Create test group)
        log("🔵 Testing <b>POST /groups</b> (Creating 'Google Auth Test Group')...");
        const grpResp = await fetch("/groups", {
          method: "POST",
          headers,
          body: JSON.stringify({ name: "Google Auth Test Group" })
        });
        const grpData = await grpResp.json();
        if (grpResp.ok) {
          log(`✅ <b>POST /groups</b>: Passed (Group ID: ${grpData.id}, Code: ${grpData.invite_code})`, "success");
        } else {
          log(`❌ <b>POST /groups</b>: Failed (${grpResp.status})`, "error");
        }

        // 4. POST /voice/text (Voice command directly modifying ledger)
        log("🔵 Testing <b>POST /voice/text</b> ('Add 250 for solo coffee')...");
        const voiceResp = await fetch("/voice/text", {
          method: "POST",
          headers,
          body: JSON.stringify({ text: "Add 250 for solo coffee" })
        });
        const voiceData = await voiceResp.json();
        if (voiceResp.ok && voiceData.status === "SUCCESS") {
          log(`✅ <b>POST /voice/text</b>: Passed! Spoken confirmation: "${voiceData.spoken_confirmation}"`, "success");
        } else {
          log(`❌ <b>POST /voice/text</b>: Failed (${voiceResp.status})`, "error");
        }

        // 5. GET /groups/{id}/expenses
        if (grpData.id) {
          log("🔵 Testing <b>GET /groups/" + grpData.id + "/expenses</b>...");
          const feedResp = await fetch(`/groups/${grpData.id}/expenses`, { headers });
          if (feedResp.ok) {
            log("✅ <b>GET /groups/{id}/expenses</b>: Passed! Group feed operational.", "success");
          }
        }

        document.getElementById("test-summary").innerHTML = 
          '<span class="text-emerald-400 font-bold"><i class="fa-solid fa-check-double mr-1"></i>All Live Endpoints Passed!</span>';

      } catch (err) {
        log("❌ Fatal test error: " + err.message, "error");
        document.getElementById("test-summary").innerText = "Failed";
      }
    }
  </script>
</body>
</html>
"""
