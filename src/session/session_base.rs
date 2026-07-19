use uuid::Uuid;

use crate::types::{ChatMessage, Role};

#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone)]
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

    #[allow(dead_code)]
    pub fn add_assistant_message(&mut self, message: &str) {
        let chat_message = ChatMessage::new(Role::Assistant, message);
        self.add_message(chat_message);
    }

    #[allow(dead_code)]
    pub fn add_tool_message(&mut self, message: &str) {
        let chat_message = ChatMessage::new(Role::Tool, message);
        self.add_message(chat_message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_session_new() {
        let session = Session::new("system message");
        assert!(!session.id.is_nil());
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].role, Role::System);
        assert_eq!(
            session.messages[0].content,
            Some("system message".to_string())
        );
    }

    #[rstest]
    #[case("system message", "user message")]
    #[case("", "user message 2")]
    #[case("", "")]
    fn test_add_user_message(#[case] system_message: &str, #[case] message: &str) {
        let mut session = Session::new(system_message);
        session.add_user_message(message);
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].role, Role::User);
        assert_eq!(session.messages[1].content, Some(message.to_string()));
    }

    #[rstest]
    #[case("system message", "assistant message")]
    #[case("", "assistant message 2")]
    #[case("", "")]
    fn test_add_assistant_message(#[case] system_message: &str, #[case] message: &str) {
        let mut session = Session::new(system_message);
        session.add_assistant_message(message);
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].role, Role::Assistant);
        assert_eq!(session.messages[1].content, Some(message.to_string()));
    }

    #[rstest]
    #[case("system message", "tool message")]
    #[case("", "tool message 2")]
    #[case("", "")]
    fn test_add_tool_message(#[case] system_message: &str, #[case] message: &str) {
        let mut session = Session::new(system_message);
        session.add_tool_message(message);
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].role, Role::Tool);
        assert_eq!(session.messages[1].content, Some(message.to_string()));
    }

    #[rstest]
    #[case(Session {
        id: Uuid::new_v4(),
        messages: vec![ChatMessage::new(Role::System, "system message"), ChatMessage::new(Role::User, "user message")]
    }, 2)]
    #[case(Session {
        id: Uuid::new_v4(),
        messages: vec![ChatMessage::new(Role::System, "system message"), ChatMessage::new(Role::User, "user message"), ChatMessage::new(Role::User, "user message 2")]
    }, 3)]
    #[case(Session {
        id: Uuid::new_v4(),
        messages: vec![ChatMessage::new(Role::System, "system message"), ChatMessage::new(Role::User, "user message"), ChatMessage::new(Role::User, "")]
    }, 3)]
    fn test_get_messages(#[case] session: Session, #[case] messages_len: usize) {
        let messages = session.get_messages();
        assert_eq!(messages.len(), messages_len);
        assert_eq!(messages[0].role, Role::System);
    }
}
