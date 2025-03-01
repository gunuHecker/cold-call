# define format of message as 
{"role" : "user", "message" : "Hello, how are you?"}

class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.messages = []
        # Add other session-related attributes here

    def call_deepggram_client(self, audio_bytes):
        # do here
        pass
    
    def add_message(self, role, message):
        self.messages.append({"role": role, "message": message})
        return f"Message added to session {self.session_id}."

class SessionManager:
    _instance = None
    _sessions = {}

    @staticmethod
    def get_instance():
        if SessionManager._instance is None:
            SessionManager._instance = SessionManager()
        return SessionManager._instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance._sessions = {}
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._sessions = {}

    def initiate_session(self, session_id):
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id)
            return f"Session {session_id} initiated."
        else:
            return f"Session {session_id} already exists."

    def get_session(self, session_id):
        if session_id in self._sessions:
            return self._sessions[session_id]
        else:
            return None
        
    def cleanup_session(self, session_id):
        if session_id in self._sessions:
            del self._sessions[session_id]
            return f"Session {session_id} cleaned up."
        else:
            return f"Session {session_id} does not exist."

