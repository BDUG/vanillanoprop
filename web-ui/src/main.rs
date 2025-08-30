use gloo_net::http::Request;
use gloo_timers::future::TimeoutFuture;
use serde::Deserialize;
use serde_json::Value;
use wasm_bindgen_futures::spawn_local;
use web_sys::HtmlSelectElement;
use yew::prelude::*;
use yew::TargetCast;

#[derive(Deserialize)]
struct JobId {
    job_id: String,
}

#[derive(Deserialize, Clone)]
struct JobInfo {
    status: String,
    progress: f32,
    result: Option<Value>,
}

async fn poll_job(job: String, job_info: UseStateHandle<Option<JobInfo>>) {
    loop {
        let resp = Request::get(&format!("/jobs/{job}/status"))
            .send()
            .await
            .ok();
        let Some(resp) = resp else { break };
        if resp.status() != 200 {
            break;
        }
        let info: JobInfo = resp.json().await.unwrap_or(JobInfo {
            status: "Failed".into(),
            progress: 0.0,
            result: None,
        });
        job_info.set(Some(info.clone()));
        if info.status == "Completed" || info.status == "Failed" {
            break;
        }
        TimeoutFuture::new(1000).await;
    }
}

#[function_component(App)]
fn app() -> Html {
    let dataset = use_state(|| "mnist".to_string());
    let model = use_state(|| "backprop".to_string());
    let job_info = use_state(|| None::<JobInfo>);

    let on_dataset_change = {
        let dataset = dataset.clone();
        Callback::from(move |e: Event| {
            let input: HtmlSelectElement = e.target_unchecked_into();
            dataset.set(input.value());
        })
    };

    let on_model_change = {
        let model = model.clone();
        Callback::from(move |e: Event| {
            let input: HtmlSelectElement = e.target_unchecked_into();
            model.set(input.value());
        })
    };

    let start_train = {
        let dataset = dataset.clone();
        let job_info = job_info.clone();
        Callback::from(move |_| {
            let ds = (*dataset).clone();
            let job_info = job_info.clone();
            spawn_local(async move {
                let body = serde_json::json!({ "dataset": ds });
                if let Ok(resp) = Request::post("/train").json(&body).unwrap().send().await {
                    if let Ok(job) = resp.json::<JobId>().await {
                        poll_job(job.job_id, job_info).await;
                    }
                }
            });
        })
    };

    let start_infer = {
        let dataset = dataset.clone();
        let job_info = job_info.clone();
        Callback::from(move |_| {
            let ds = (*dataset).clone();
            let job_info = job_info.clone();
            spawn_local(async move {
                let body = serde_json::json!({ "dataset": ds });
                if let Ok(resp) = Request::post("/infer").json(&body).unwrap().send().await {
                    if let Ok(job) = resp.json::<JobId>().await {
                        poll_job(job.job_id, job_info).await;
                    }
                }
            });
        })
    };

    html! {
        <div>
            <h1>{ "VanillaNoProp UI" }</h1>
            <div>
                <label>{ "Dataset: " }</label>
                <select onchange={on_dataset_change}>
                    <option value="mnist" selected={(*dataset) == "mnist"}>{ "MNIST" }</option>
                    <option value="cifar10" selected={(*dataset) == "cifar10"}>{ "CIFAR-10" }</option>
                </select>
            </div>
            <div>
                <label>{ "Model Variant: " }</label>
                <select onchange={on_model_change}>
                    <option value="backprop" selected={(*model) == "backprop"}>{ "Backprop" }</option>
                    <option value="noprop" selected={(*model) == "noprop"}>{ "NoProp" }</option>
                </select>
            </div>
            <div>
                <button onclick={start_train}>{ "Start Training" }</button>
                <button onclick={start_infer}>{ "Run Inference" }</button>
            </div>
            <div>
                <p>{ format!("Status: {}", job_info.as_ref().map(|j| j.status.clone()).unwrap_or_else(|| "Idle".into())) }</p>
                <p>{ format!("Progress: {:.0}%", job_info.as_ref().map(|j| j.progress * 100.0).unwrap_or(0.0)) }</p>
                {
                    if let Some(info) = &*job_info {
                        if let Some(res) = &info.result {
                            html! { <pre>{ serde_json::to_string_pretty(res).unwrap_or_default() }</pre> }
                        } else {
                            Html::default()
                        }
                    } else {
                        Html::default()
                    }
                }
            </div>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
