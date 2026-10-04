import "./style.css";
import { formatMessage } from "./format_message.js";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const errorMessage = document.querySelector("#error-message");
const clearSessionButton = document.querySelector("#clear-session");
const conversationHistory = [];
const MAX_HISTORY_TURN_LENGTH = 4000;
const WELCOME_MESSAGE =
  "Hallo! 👋 Ich bin dein Krank Demo-Chatbot. Wie kann ich dir helfen?";
let sessionId = crypto.randomUUID();

function addMessage(text, sender, { formatted = false } = {}) {
  const article = document.createElement("article");
  article.className = `message ${sender}-message`;

  const content = document.createElement("div");
  content.className = "message-content";
  if (formatted) {
    content.innerHTML = formatMessage(text);
  } else {
    content.textContent = text;
  }
  const time = document.createElement("time");
  time.textContent = new Intl.DateTimeFormat("de-DE", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());

  article.append(content, time);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function createLoadingIndicator() {
  const article = document.createElement("article");
  article.className = "message bot-message loading-message";
  article.setAttribute("role", "status");
  const spinner = document.createElement("span");
  spinner.className = "loading-spinner";
  spinner.setAttribute("aria-hidden", "true");
  const label = document.createElement("span");
  label.textContent = "Ich prüfe deine Frage …";
  article.append(spinner, label);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return { article, label };
}

function clearSession() {
  conversationHistory.length = 0;
  sessionId = crypto.randomUUID();
  errorMessage.textContent = "";
  messages.replaceChildren();
  addMessage(WELCOME_MESSAGE, "bot");
  input.value = "";
  input.focus();
}

clearSessionButton.addEventListener("click", clearSession);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  errorMessage.textContent = "";
  addMessage(message, "user");
  const loading = createLoadingIndicator();
  let streamedReply = null;
  let streamedArticle = null;
  input.value = "";
  input.disabled = true;
  clearSessionButton.disabled = true;

  try {
    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history: conversationHistory,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(
        error.detail || `Der Server hat einen Fehler gemeldet (${response.status}).`,
      );
    }

    if (!response.body) {
      throw new Error("Der Server hat keinen Ereignis-Stream geliefert.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let replyAdded = false;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const records = buffer.split("\n\n");
      buffer = records.pop() || "";
      for (const record of records) {
        const dataLine = record.split("\n").find((line) => line.startsWith("data: "));
        if (!dataLine) continue;
        const agentEvent = JSON.parse(dataLine.slice(6));
        if (agentEvent.type === "status") {
          loading.label.textContent = agentEvent.message;
        } else if (agentEvent.type === "answer_delta") {
          if (!streamedReply) {
            loading.article.remove();
            streamedArticle = addMessage("", "bot");
            streamedReply = streamedArticle.querySelector(".message-content");
          }
          streamedReply.textContent += agentEvent.text;
          messages.scrollTop = messages.scrollHeight;
        } else if (agentEvent.type === "final") {
          if (streamedReply) {
            streamedReply.innerHTML = formatMessage(agentEvent.reply);
          } else {
            loading.article.remove();
            addMessage(agentEvent.reply, "bot", { formatted: true });
          }
          conversationHistory.push(
            { role: "user", content: message },
            {
              role: "assistant",
              content: agentEvent.reply.slice(0, MAX_HISTORY_TURN_LENGTH),
            },
          );
          if (conversationHistory.length > 20) {
            conversationHistory.splice(0, conversationHistory.length - 20);
          }
          replyAdded = true;
        } else if (agentEvent.type === "error") {
          loading.article.remove();
          streamedArticle?.remove();
          errorMessage.textContent = agentEvent.message;
        }
      }
      if (done) break;
    }
    if (!replyAdded && !errorMessage.textContent) {
      throw new Error("Der Agent hat keine Antwort zurückgegeben.");
    }
  } catch (error) {
    loading.article.remove();
    streamedArticle?.remove();
    errorMessage.textContent =
      error instanceof Error
        ? `${error.message} Läuft das Backend auf Port 8000?`
        : "Die Nachricht konnte nicht gesendet werden.";
  } finally {
    input.disabled = false;
    clearSessionButton.disabled = false;
    input.focus();
  }
});
