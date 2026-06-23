use uuid::Uuid;

use crate::types::{ChatMessage, Role};

pub struct Session {
    pub id: Uuid,
    messages: Vec<ChatMessage>,
}

impl Session {
    pub fn new(system_message: &str) -> Self {
        let id = Uuid::new_v4();
        Self {
            id,
            messages: vec![ChatMessage::new(Role::System, system_message)],
        }
    }

    pub fn get_messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn add_message(&mut self, chat_message: ChatMessage) {
        self.messages.push(chat_message);
    }

    pub fn add_user_message(&mut self, message: &str) {
        let chat_message = ChatMessage::new(Role::User, message);
        self.add_message(chat_message);
    }

    pub fn add_assistant_message(&mut self, message: &str) {
        let chat_message = ChatMessage::new(Role::Assistant, message);
        self.add_message(chat_message);
    }

    pub fn add_tool_message(&mut self, message: &str) {
        let chat_message = ChatMessage::new(Role::Tool, message);
        self.add_message(chat_message);
    }
}
