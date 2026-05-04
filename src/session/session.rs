use std::fmt::{self, Display, Formatter};

use serde::Serialize;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Display for Role {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    role: Role,
    content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

pub struct Session {
    pub id: Uuid,
    messages: Vec<ChatMessage>,
}

impl Session {
    pub fn new(system_message: impl Into<String>) -> Self {
        let id = Uuid::new_v4();
        Self {
            id,
            messages: vec![ChatMessage::new(Role::System, system_message)],
        }
    }

    pub fn get_messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn add_message(&mut self, role: Role, content: impl Into<String>) {
        self.messages.push(ChatMessage::new(role, content));
    }

    pub fn add_user_message(&mut self, message: impl Into<String>) {
        self.add_message(Role::User, message);
    }

    pub fn add_assistant_message(&mut self, message: impl Into<String>) {
        self.add_message(Role::Assistant, message);
    }

    pub fn add_tool_message(&mut self, message: impl Into<String>) {
        self.add_message(Role::Tool, message);
        use crate::session::SessionManager;
    }
}
