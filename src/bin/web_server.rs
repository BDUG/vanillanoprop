use actix_web::{web, App, HttpResponse, HttpServer};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::wrappers::BroadcastStream;
use vanillanoprop::config::Config;
use vanillanoprop::data::DatasetKind;
use vanillanoprop::job::{JobRegistry, JobStatus};

mod train_backprop;

#[derive(Deserialize)]
struct DatasetParam {
    dataset: Option<String>,
}

#[derive(Serialize)]
struct JobId {
    job_id: String,
}

fn parse_dataset(
    query: &web::Query<DatasetParam>,
    body: &Option<web::Json<DatasetParam>>,
) -> Result<DatasetKind, HttpResponse> {
    let name = body
        .as_ref()
        .and_then(|j| j.dataset.clone())
        .or_else(|| query.dataset.clone())
        .ok_or_else(|| HttpResponse::BadRequest().body("unsupported dataset"))?;
    DatasetKind::from_str(&name)
        .ok_or_else(|| HttpResponse::BadRequest().body("unsupported dataset"))
}

async fn train(
    registry: web::Data<JobRegistry>,
    query: web::Query<DatasetParam>,
    body: Option<web::Json<DatasetParam>>,
) -> HttpResponse {
    let dataset = match parse_dataset(&query, &body) {
        Ok(ds) => ds,
        Err(resp) => return resp,
    };
    let job_id = registry.create_job();
    let job_id_resp = job_id.clone();
    let reg = registry.clone();
    tokio::spawn({
        let reg = reg.clone();
        let job_id = job_id.clone();
        async move {
            reg.set_status(&job_id, JobStatus::Running);
            let config = Config::default();
            let ds = dataset;
            let reg2 = reg.clone();
            let job_id2 = job_id.clone();
            let res = tokio::task::spawn_blocking(move || {
                train_backprop::run(ds, "sgd", false, 1, None, None, &config, None, None);
            })
            .await;
            match res {
                Ok(_) => {
                    reg2.update_progress(&job_id2, 1.0);
                    reg2.set_status(&job_id2, JobStatus::Completed);
                    reg2.set_result(&job_id2, json!({"message": "training complete"}));
                }
                Err(e) => {
                    reg2.set_status(&job_id2, JobStatus::Failed);
                    reg2.set_result(&job_id2, json!({"error": format!("{e}")}));
                }
            }
        }
    });
    HttpResponse::Ok().json(JobId { job_id: job_id_resp })
}

async fn infer(
    registry: web::Data<JobRegistry>,
    query: web::Query<DatasetParam>,
    body: Option<web::Json<DatasetParam>>,
) -> HttpResponse {
    let dataset = match parse_dataset(&query, &body) {
        Ok(ds) => ds,
        Err(resp) => return resp,
    };
    let job_id = registry.create_job();
    let job_id_resp = job_id.clone();
    let reg = registry.clone();
    tokio::spawn({
        let reg = reg.clone();
        let job_id = job_id.clone();
        async move {
            reg.set_status(&job_id, JobStatus::Running);
            let ds = dataset;
            let reg2 = reg.clone();
            let job_id2 = job_id.clone();
            let res = tokio::task::spawn_blocking(move || {
                vanillanoprop::predict::run(ds, None, false, 0)
            })
            .await;
            match res {
                Ok(pred) => {
                    reg2.update_progress(&job_id2, 1.0);
                    reg2.set_status(&job_id2, JobStatus::Completed);
                    reg2.set_result(&job_id2, pred);
                }
                Err(e) => {
                    reg2.set_status(&job_id2, JobStatus::Failed);
                    reg2.set_result(&job_id2, json!({"error": format!("{e}")}));
                }
            }
        }
    });
    HttpResponse::Ok().json(JobId { job_id: job_id_resp })
}

async fn job_status(
    registry: web::Data<JobRegistry>,
    id: web::Path<String>,
) -> HttpResponse {
    match registry.get_info(&id) {
        Some(info) => HttpResponse::Ok().json(info),
        None => HttpResponse::NotFound().finish(),
    }
}

async fn job_result(
    registry: web::Data<JobRegistry>,
    id: web::Path<String>,
) -> HttpResponse {
    match registry.get_result(&id) {
        Some(res) => HttpResponse::Ok().json(res),
        None => HttpResponse::NotFound().finish(),
    }
}

async fn job_events(
    registry: web::Data<JobRegistry>,
    id: web::Path<String>,
) -> HttpResponse {
    if let Some(rx) = registry.subscribe(&id) {
        let stream = BroadcastStream::new(rx).filter_map(|msg| async move {
            match msg {
                Ok(m) => Some(Ok::<_, actix_web::Error>(web::Bytes::from(format!("data: {}\n\n", m)))),
                Err(_) => None,
            }
        });
        HttpResponse::Ok()
            .insert_header(("Content-Type", "text/event-stream"))
            .streaming(stream)
    } else {
        HttpResponse::NotFound().finish()
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    let registry = web::Data::new(JobRegistry::new());
    HttpServer::new(move || {
        App::new()
            .app_data(registry.clone())
            .route("/train", web::post().to(train))
            .route("/infer", web::post().to(infer))
            .route("/jobs/{id}/status", web::get().to(job_status))
            .route("/jobs/{id}/result", web::get().to(job_result))
            .route("/jobs/{id}/events", web::get().to(job_events))
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
