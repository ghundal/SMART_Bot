"""
Implements a ChatHistoryManager class that handles saving, retrieving, and
deleting chat records in a PostgreSQL database.
"""

import json
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import text

from .database import SessionLocal


class ChatHistoryManager:
    def __init__(self, model):
        """Initialize the chat history manager with PostgreSQL connection"""
        self.model = model

        # Connect to PostgreSQL
        self.SessionLocal = SessionLocal

    def save_chat(self, chat_to_save: Dict, user_email: str, session_id: str) -> None:
        """Save a chat to PostgreSQL database"""
        try:
            # Create a new database session
            db_session = self.SessionLocal()

            # Convert messages to JSON string
            messages_json = json.dumps(chat_to_save["messages"], ensure_ascii=False)

            # Check if chat exists and update or insert accordingly
            sql = """
            SELECT COUNT(*) FROM chat_history
            WHERE chat_id = :chat_id AND user_email = :user_email AND model = :model
            """

            result = db_session.execute(
                text(sql),
                {
                    "chat_id": chat_to_save["chat_id"],
                    "user_email": user_email,
                    "model": self.model,
                },
            ).scalar()

            if result > 0:
                # Update existing chat
                sql = """
                UPDATE chat_history
                SET title = :title,
                    messages = :messages,
                    dts = :dts,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chat_id = :chat_id AND session_id = :session_id AND model = :model
                """
            else:
                # Insert new chat
                sql = """
                INSERT INTO chat_history (
                    chat_id, session_id, user_email, model, title, messages, dts, created_at, updated_at
                ) VALUES (
                    :chat_id, :session_id, :user_email, :model, :title, :messages, :dts, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                """

            # Execute the query
            db_session.execute(
                text(sql),
                {
                    "chat_id": chat_to_save["chat_id"],
                    "session_id": session_id,
                    "user_email": user_email,
                    "model": self.model,
                    "title": chat_to_save.get("title", "Untitled Chat"),
                    "messages": messages_json,
                    "dts": chat_to_save.get("dts", int(datetime.now().timestamp())),
                },
            )

            # Commit the changes
            db_session.commit()

        except Exception as e:
            db_session.rollback()
            print(f"Error saving chat {chat_to_save['chat_id']}: {str(e)}")
            traceback.print_exc()
            raise e
        finally:
            db_session.close()

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get a specific chat by ID from the database"""
        try:
            # Create a new database session
            db_session = self.SessionLocal()

            # Query the database
            sql = """
            SELECT chat_id, session_id, model, title, messages, dts, created_at, updated_at
            FROM chat_history
            WHERE chat_id = :chat_id
            """

            result = db_session.execute(
                text(sql),
                {"chat_id": chat_id},
            ).fetchone()

            if result:
                # Convert to dictionary
                chat_data = {
                    "chat_id": result[0],
                    "session_id": result[1],
                    "model": result[2],
                    "title": result[3],
                    "messages": (
                        result[4] if isinstance(result[4], list) else json.loads(result[4])
                    ),
                    "dts": result[5],
                    "created_at": result[6].isoformat() if result[6] else None,
                    "updated_at": result[7].isoformat() if result[7] else None,
                }
                return chat_data
            else:
                return None

        except Exception as e:
            print(f"Error retrieving chat {chat_id}: {str(e)}")
            traceback.print_exc()
            return None
        finally:
            db_session.close()

    def get_recent_chats(self, user_email: str, limit: Optional[int] = None) -> List[Dict]:
        """Get recent chats from the database, optionally limited to a specific number"""
        recent_chats = []
        try:
            # Create a new database session
            db_session = self.SessionLocal()

            # Query the database
            sql = """
            SELECT chat_id, session_id, model, title, messages, dts, created_at, updated_at
            FROM chat_history
            WHERE user_email = :user_email AND model = :model
            ORDER BY dts DESC
            """

            if limit:
                sql += f" LIMIT {int(limit)}"

            results = db_session.execute(
                text(sql), {"user_email": user_email, "model": self.model}
            ).fetchall()

            for result in results:
                # Convert to dictionary
                chat_data = {
                    "chat_id": result[0],
                    "session_id": result[1],
                    "model": result[2],
                    "title": result[3],
                    "messages": (
                        result[4] if isinstance(result[4], list) else json.loads(result[4])
                    ),
                    "dts": result[5],
                    "created_at": result[6].isoformat() if result[6] else None,
                    "updated_at": result[7].isoformat() if result[7] else None,
                }
                recent_chats.append(chat_data)

            return recent_chats

        except Exception as e:
            print(f"Error retrieving recent chats: {str(e)}")
            traceback.print_exc()
            return []
        finally:
            db_session.close()

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat from the database"""
        try:
            # Create a new database session
            db_session = self.SessionLocal()

            # Delete from database
            sql = """
            DELETE FROM chat_history
            WHERE chat_id = :chat_id
            """

            result = db_session.execute(
                text(sql),
                {"chat_id": chat_id},
            )
            # Check if any row was actually deleted
            rows_affected = result.rowcount
            if rows_affected == 0:
                return False

            # Commit the changes
            db_session.commit()

            return True

        except Exception as e:
            db_session.rollback()
            print(f"Error deleting chat {chat_id}: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            db_session.close()
