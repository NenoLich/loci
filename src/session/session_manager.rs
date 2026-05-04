use uuid::Uuid;

use crate::session::Session;

pub struct SessionManager {
    sessions: Vec<Session>
}

impl SessionManager {
    pub fn new() -> Self {
        Self { sessions: vec![] }
    }

    pub fn start_session(&mut self, system_prompt: impl Into<String>) -> &mut Session {
        let session = Session::new(system_prompt);
        self.sessions.push(session);
        self.sessions.last_mut().unwrap()
    }

    pub fn get_session_mut(&mut self, id: Uuid) -> Option<&mut Session> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }
}