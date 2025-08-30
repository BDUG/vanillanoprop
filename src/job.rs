use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Serialize};
use serde_json::Value;
use tokio::sync::broadcast;
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
    tx: broadcast::Sender<String>,
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
        let (tx, _rx) = broadcast::channel(100);
        let job = InternalJob {
            info: JobInfo {
                status: JobStatus::Queued,
                progress: 0.0,
                result: None,
            },
            tx,
        };
        self.inner.lock().unwrap().insert(id.clone(), job);
        id
    }

    pub fn set_status(&self, id: &str, status: JobStatus) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(job) = inner.get_mut(id) {
            job.info.status = status.clone();
            let _ = job.tx.send(serde_json::json!({"status": status}).to_string());
        }
    }

    pub fn update_progress(&self, id: &str, progress: f32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(job) = inner.get_mut(id) {
            job.info.progress = progress;
            let _ = job.tx.send(serde_json::json!({"progress": progress}).to_string());
        }
    }

    pub fn set_result(&self, id: &str, result: Value) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(job) = inner.get_mut(id) {
            job.info.result = Some(result.clone());
            let _ = job.tx.send(serde_json::json!({"result": result}).to_string());
        }
    }

    pub fn get_info(&self, id: &str) -> Option<JobInfo> {
        self.inner.lock().unwrap().get(id).map(|j| j.info.clone())
    }

    pub fn get_result(&self, id: &str) -> Option<Value> {
        self.inner.lock().unwrap().get(id).and_then(|j| j.info.result.clone())
    }

    pub fn subscribe(&self, id: &str) -> Option<broadcast::Receiver<String>> {
        self.inner.lock().unwrap().get(id).map(|j| j.tx.subscribe())
    }
}
