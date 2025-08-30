use actix_web::{web, App, HttpResponse, HttpServer};
use serde::Deserialize;
use vanillanoprop::config::Config;
use vanillanoprop::data::DatasetKind;

mod train_backprop;

#[derive(Deserialize)]
struct DatasetParam {
    dataset: Option<String>,
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
    query: web::Query<DatasetParam>,
    body: Option<web::Json<DatasetParam>>,
) -> HttpResponse {
    let dataset = match parse_dataset(&query, &body) {
        Ok(ds) => ds,
        Err(resp) => return resp,
    };
    let config = Config::default();
    let ds = dataset;
    let _ = web::block(move || {
        train_backprop::run(ds, "sgd", false, 1, None, None, &config, None, None);
    })
    .await;
    HttpResponse::Ok().body("training complete")
}

async fn infer(
    query: web::Query<DatasetParam>,
    body: Option<web::Json<DatasetParam>>,
) -> HttpResponse {
    let dataset = match parse_dataset(&query, &body) {
        Ok(ds) => ds,
        Err(resp) => return resp,
    };
    let ds = dataset;
    let _ = web::block(move || {
        vanillanoprop::predict::run(ds, None, false, 0);
    })
    .await;
    HttpResponse::Ok().body("inference complete")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/train", web::post().to(train)).route("/infer", web::post().to(infer)))
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}
