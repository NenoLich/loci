use uuid::Uuid;

use crate::session::Session;

#[derive(Default)]
pub struct SessionManager {
    sessions: Vec<Session>,
}

impl SessionManager {
    pub fn start_session(&mut self, system_prompt: &str) -> &mut Session {
        let session = Session::new(system_prompt);
        self.sessions.push(session);
        self.sessions.last_mut().unwrap()
    }

    #[allow(dead_code)]
    pub fn get_session_mut(&mut self, id: Uuid) -> Option<&mut Session> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_manager() {
        let mut manager = SessionManager::default();
        let session = manager.start_session("Hello");
        session.add_user_message("World");
        let id = session.id; // Copy
        assert_eq!(session.get_messages().len(), 2);

        let session_lookup = manager.get_session_mut(id).unwrap();
        assert_eq!(
            session_lookup.get_messages()[0].content.as_deref(),
            Some("Hello")
        );
        assert_eq!(
            session_lookup.get_messages()[1].content.as_deref(),
            Some("World")
        );
    }

    #[test]
    fn test_get_session_mut_none() {
        let mut manager = SessionManager::default();
        let result = manager.get_session_mut(Uuid::new_v4());
        assert!(result.is_none());
    }

    #[test]
    fn test_multiple_sessions() {
        let mut manager = SessionManager::default();
        let s1 = manager.start_session("sys1");
        s1.add_user_message("msg1");
        let id1 = s1.id;
        let s2 = manager.start_session("sys2");
        s2.add_user_message("msg2");
        let id2 = s2.id;

        assert_eq!(
            manager.get_session_mut(id1).unwrap().get_messages().len(),
            2
        );
        assert_eq!(
            manager.get_session_mut(id2).unwrap().get_messages().len(),
            2
        );
        assert_ne!(id1, id2);
    }
}
