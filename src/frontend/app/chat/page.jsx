'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import ProtectedRoute from '../../components/auth/ProtectedRoute';
import DataService from '../../services/DataService';
import styles from '../../components/chat/chat.module.css';

function ChatContent() {
  // States for chat functionality
  const [chatId, setChatId] = useState(null);
  const [hasActiveChat, setHasActiveChat] = useState(false);
  const [chat, setChat] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [recentChats, setRecentChats] = useState([]);
  const [showSidebar, setShowSidebar] = useState(true);
  const [model, setModel] = useState('gemma3:12b');
  const messagesEndRef = useRef(null);

  const router = useRouter();
  const searchParams = useSearchParams();

  // Extract chat_id and model from URL parameters
  useEffect(() => {
    const id = searchParams.get('id');
    const modelParam = searchParams.get('model') || 'gemma3:12b';

    if (id) {
      setChatId(id);
      setHasActiveChat(true);
      fetchChat(id);
    } else {
      setChatId(null);
      setHasActiveChat(false);
      setChat(null);
    }

    setModel(modelParam);
  }, [searchParams]);

  // Fetch recent chat history on component mount
  useEffect(() => {
    fetchRecentChats();
  }, []);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [chat]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Fetch functions using DataService
  const fetchRecentChats = async () => {
    try {
      const response = await DataService.GetChats();
      setRecentChats(response.data);
    } catch (error) {
      console.error('Error fetching recent chats:', error);
    }
  };

  const fetchChat = async (id) => {
    try {
      setChat(null);
      const response = await DataService.GetChat(model, id);
      setChat(response.data);
    } catch (error) {
      console.error('Error fetching chat:', error);
      setChat(null);
      setHasActiveChat(false);
    }
  };

  const startNewChat = async (message) => {
    try {
      setIsTyping(true);
      setHasActiveChat(true);

      // Show temporary user message while waiting for response
      setChat({
        messages: [
          {
            message_id: DataService.uuid(),
            role: 'user',
            content: message,
          },
        ],
      });

      // Make a POST request to create a new chat
      const response = await DataService.StartChatWithLLM(model, { content: message });

      if (!response || !response.data) {
        throw new Error('Failed to create new chat - no response data');
      }

      setChat(response.data);
      setChatId(response.data.chat_id);
      router.push(`/chat?model=${model}&id=${response.data.chat_id}`);
      fetchRecentChats(); // Refresh the sidebar
    } catch (error) {
      console.error('Error starting new chat:', error);
      setIsTyping(false);
      setHasActiveChat(false);
      // Alert the user
      alert(`Failed to start new chat: ${error.message}`);
    } finally {
      setIsTyping(false);
    }
  };

  const continueChat = async (message) => {
    if (!chatId) {
      return;
    }

    try {
      setIsTyping(true);

      // Add temporary user message
      const updatedChat = { ...chat };
      updatedChat.messages.push({
        message_id: DataService.uuid(),
        role: 'user',
        content: message,
      });
      setChat(updatedChat);

      const response = await DataService.ContinueChatWithLLM(model, chatId, { content: message });
      setChat(response.data);
      fetchRecentChats(); // Refresh the sidebar
    } catch (error) {
      console.error('Error continuing chat:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const deleteChat = async (id) => {
    try {
      await DataService.DeleteChat(id);
      fetchRecentChats();
      if (chatId === id) {
        setChat(null);
        setChatId(null);
        setHasActiveChat(false);
        router.push(`/chat?model=${model}`);
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) {
      return;
    }

    if (hasActiveChat && chatId) {
      continueChat(inputMessage);
    } else {
      startNewChat(inputMessage);
    }

    setInputMessage('');
  };

  const handleSelectChat = (id) => {
    router.push(`/chat?model=${model}&id=${id}`);
  };

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    setModel(newModel);

    // Update URL with new model
    if (chatId) {
      router.push(`/chat?model=${newModel}&id=${chatId}`);
    } else {
      router.push(`/chat?model=${newModel}`);
    }
  };

  // New function to handle query suggestions properly
  const handleSuggestionClick = (suggestion) => {
    if (hasActiveChat && chatId) {
      continueChat(suggestion);
    } else {
      startNewChat(suggestion);
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  return (
    <div className={styles.chatContainer}>
      {/* Model Selector Button - Floating Fixed Position */}
      <div className={styles.floatingModelSelector}>
        <span className={styles.modelLabel}>Model:</span>
        <select
          id="model-select"
          value={model}
          onChange={handleModelChange}
          className={styles.modelSelectButton}
        >
          <option value="llama3:8b">Llama</option>
          <option value="gemma3:12b">Gemma</option>
        </select>
      </div>

      {/* Sidebar */}
      <div className={`${styles.sidebar} ${showSidebar ? '' : styles.hidden}`}>
        <div className={styles.sidebarHeader}>
          <h2>Recent Chats</h2>
          <button
            className={styles.newChatButton}
            onClick={() => {
              setChat(null);
              setChatId(null);
              setHasActiveChat(false);
              router.push(`/chat?model=${model}`);
            }}
          >
            + New Chat
          </button>
        </div>

        <div className={styles.chatList}>
          {recentChats.length > 0 ? (
            recentChats.map((item) => (
              <div
                key={item.chat_id}
                className={`${styles.chatItem} ${item.chat_id === chatId ? styles.active : ''}`}
                onClick={() => handleSelectChat(item.chat_id)}
              >
                <div className={styles.chatItemContent}>
                  <div className={styles.chatItemTitle}>{item.title}</div>
                  <div className={styles.chatItemDate}>{formatDate(item.dts)}</div>
                </div>
                <button
                  className={styles.deleteButton}
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChat(item.chat_id);
                  }}
                >
                  ×
                </button>
              </div>
            ))
          ) : (
            <div className={styles.noChats}>No chats yet</div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className={styles.mainContent}>
        <div className={styles.chatHeader}>
          <button
            className={styles.toggleSidebarButton}
            onClick={() => setShowSidebar(!showSidebar)}
          >
            {showSidebar ? '<<' : '>>'}
          </button>
        </div>

        <div className={styles.messagesContainer}>
          {hasActiveChat && chat?.messages ? (
            <div className={styles.messages}>
              {chat.messages.map((message) => (
                <div
                  key={message.message_id}
                  className={`${styles.message} ${
                    message.role === 'assistant' ? styles.assistant : styles.user
                  }`}
                >
                  <div className={styles.messageContent}>
                    {message.content.split('\n').map((line, i) => (
                      <p key={i}>{line}</p>
                    ))}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className={`${styles.message} ${styles.assistant}`}>
                  <div className={styles.messageContent}>
                    <div className={styles.typingIndicator}>
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          ) : !hasActiveChat ? (
            <div className={styles.welcomeContainer}>
              <h1 className={styles.welcomeTitle}>SMART Chat Assistant</h1>
              <p className={styles.welcomeText}>
                Ask me any questions about your documents, and I'll help you find the information
                you need.
              </p>
              <div className={styles.exampleQueries}>
                <h3>Example queries:</h3>
                <ul>
                  <li onClick={() => handleSuggestionClick('Was ist ein zufälliger Wald?')}>
                    'Was ist ein zufälliger Wald?'
                  </li>
                  <li onClick={() => handleSuggestionClick('What is a random forest?')}>
                    What is a random forest?
                  </li>
                  <li
                    onClick={() =>
                      handleSuggestionClick('Summarize the key concepts from lecture on CNNs')
                    }
                  >
                    Summarize the key concepts from lecture on CNNs
                  </li>
                </ul>
              </div>
            </div>
          ) : (
            <div className={styles.loading}>Loading chat...</div>
          )}
        </div>

        <div className={styles.inputContainer}>
          <form onSubmit={handleSubmit}>
            <div className={styles.inputWrapper}>
              <textarea
                className={styles.input}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask a question about your documents..."
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                onInput={(e) => {
                  // Auto-resize textarea
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(200, e.target.scrollHeight) + 'px';
                }}
              />
              <button
                type="submit"
                className={styles.sendButton}
                disabled={isTyping || !inputMessage.trim()}
              >
                <svg
                  viewBox="0 0 24 24"
                  width="24"
                  height="24"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </div>
          </form>

          {/* Quick suggestion buttons - Add these for common follow-up questions */}
          {hasActiveChat && chat?.messages && (
            <div className={styles.suggestionContainer}>
              <button
                className={styles.suggestionButton}
                onClick={() => handleSuggestionClick('give me its assumptions')}
                disabled={isTyping}
              >
                give me assumptions
              </button>
              <button
                className={styles.suggestionButton}
                onClick={() => handleSuggestionClick('explain it in more detail')}
                disabled={isTyping}
              >
                explain in more detail
              </button>
              <button
                className={styles.suggestionButton}
                onClick={() => handleSuggestionClick('provide me with an example of it')}
                disabled={isTyping}
              >
                provide an example
              </button>
            </div>
          )}

          {chat?.top_documents && (
            <div className={styles.sourcesContainer}>
              <div className={styles.sourcesHeader}>Sources:</div>
              <div className={styles.sourcesList}>
                {chat.top_documents.map((doc, index) => (
                  <div key={index} className={styles.sourceItem}>
                    <span className={styles.sourceTitle}>
                      {doc.title || doc.class_name || 'Document'}
                    </span>
                    {doc.authors && <span className={styles.sourceAuthor}>by {doc.authors}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <ProtectedRoute>
      <ChatContent />
    </ProtectedRoute>
  );
}
