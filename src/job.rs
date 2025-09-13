use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Sender, Receiver};

use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

#[derive(Clone, Serialize)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Serialize)]
pub struct JobInfo {
    pub status: JobStatus,
    pub progress: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
}

struct InternalJob {
    info: JobInfo,
    subscribers: Vec<Sender<String>>,
}

#[derive(Clone)]
pub struct JobRegistry {
    inner: Arc<Mutex<HashMap<String, InternalJob>>>,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashMap::new())) }
    }

    pub fn create_job(&self) -> String {
        let id = Uuid::new_v4().to_string();
        let job = InternalJob {
            info: JobInfo {
                status: JobStatus::Queued,
                progress: 0.0,
                result: None,
            },
            subscribers: Vec::new(),
        };
        self.inner.lock().unwrap().insert(id.clone(), job);
        id
    }

    pub fn set_status(&self, id: &str, status: JobStatus) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(job) = inner.get_mut(id) {
            job.info.status = status.clone();
            let message = serde_json::json!({"status": status}).to_string();
            job.subscribers.retain(|tx| tx.send(message.clone()).is_ok());
        }
    }

    pub fn update_progress(&self, id: &str, progress: f32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(job) = inner.get_mut(id) {
            job.info.progress = progress;
            let message = serde_json::json!({"progress": progress}).to_string();
            job.subscribers.retain(|tx| tx.send(message.clone()).is_ok());
        }
    }

    pub fn set_result(&self, id: &str, result: Value) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(job) = inner.get_mut(id) {
            job.info.result = Some(result.clone());
            let message = serde_json::json!({"result": result}).to_string();
            job.subscribers.retain(|tx| tx.send(message.clone()).is_ok());
        }
    }

    pub fn get_info(&self, id: &str) -> Option<JobInfo> {
        self.inner.lock().unwrap().get(id).map(|j| j.info.clone())
    }

    pub fn get_result(&self, id: &str) -> Option<Value> {
        self.inner.lock().unwrap().get(id).and_then(|j| j.info.result.clone())
    }

    pub fn subscribe(&self, id: &str) -> Option<Receiver<String>> {
        let (tx, rx) = mpsc::channel();
        let mut inner = self.inner.lock().unwrap();
        if let Some(job) = inner.get_mut(id) {
            job.subscribers.push(tx);
            Some(rx)
        } else {
            None
        }
    }
}
