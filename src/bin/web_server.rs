use actix_web::{web, App, HttpResponse, HttpServer};
use vanillanoprop::config::Config;

mod train_backprop;

async fn train() -> HttpResponse {
    let config = Config::default();
    let _ = web::block(move || {
        train_backprop::run("sgd", false, 1, None, None, &config, None, None);
    })
    .await;
    HttpResponse::Ok().body("training complete")
}

async fn infer() -> HttpResponse {
    let _ = web::block(|| {
        vanillanoprop::predict::run(None, false, 0);
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
