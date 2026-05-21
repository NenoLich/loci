use uuid::Uuid;

use crate::api::types::{ChatMessage, Role};

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
    }
}
