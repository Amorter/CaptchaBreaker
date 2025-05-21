use crate::model::Model;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::Session;
use reqwest::Url;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::{env, fs};
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use std::time::Duration;

pub enum ModelLoader {
    DefaultModelLoader,
    CustomModelLoader(Box<dyn ModelLoaderTrait>),
}
pub trait ModelLoaderTrait {
    fn load(&self, model: Model) -> Result<Session, Box<dyn Error>> {
        self.load_with_execution_providers(model, vec![CPUExecutionProvider::default().build()])
    }
    fn load_with_execution_providers(&self, model: Model, providers: Vec<ExecutionProviderDispatch>) -> Result<Session, Box<dyn Error>>;
}

#[derive(Default)]
pub struct DefaultModelLoader;

fn load_one_model(path: &Path, url: &Url, providers: impl IntoIterator<Item = ExecutionProviderDispatch>) -> Result<Session, Box<dyn Error>> {
    let model_bytes: Vec<u8>;

    if !path.exists() {
        println!("Model not found locally. Downloading from {} to {}...", url.as_str(), path.display());

        // 检查是否已在运行时中
        model_bytes = if tokio::runtime::Handle::try_current().is_ok() {
            // 已在Tokio运行时中，使用现有运行时
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    download_model_streaming_internal(url, path).await
                })
            })?
        } else {
            // 不在运行时中，创建新运行时
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(async {
                download_model_streaming_internal(url, path).await
            })?
        };
        println!("Model downloaded successfully to {}.", path.display());
    } else {
        println!("Loading model from local cache: {}.", path.display());
        model_bytes = fs::read(path)?; // std::fs::read
    }
    Ok(Session::builder()?
        .with_execution_providers(providers)?
        .commit_from_memory(model_bytes.as_ref())?)
}

// 异步流式下载函数
async fn download_model_streaming_internal(url: &Url, file_path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let client = reqwest::Client::builder()
        .user_agent("CaptchaBreaker")
        .timeout(Duration::from_secs(3600)) // 1h超时
        .build()?;

    let mut response = client.get(url.clone()).send().await?;

    if !response.status().is_success() {
        return Err(format!("Request failed with status: {}", response.status()).into());
    }

    // 确保目标目录存在
    if let Some(parent_dir) = file_path.parent() {
        if !parent_dir.exists() {
            tokio::fs::create_dir_all(parent_dir).await?;
        }
    }

    // 使用临时文件下载，成功后再重命名，避免下载中断导致文件损坏
    let temp_file_path_str = format!("{}.tmp", file_path.to_string_lossy());
    let temp_file_path = Path::new(&temp_file_path_str);

    let mut dest_file = TokioFile::create(&temp_file_path).await?;
    let mut downloaded_bytes: u64 = 0;
    let total_size = response.content_length().unwrap_or(0);
    let mut all_bytes_for_session = Vec::new();

    println!("Starting streaming download...");
    while let Some(chunk) = response.chunk().await? {
        dest_file.write_all(&chunk).await?;
        all_bytes_for_session.extend_from_slice(&chunk);
        downloaded_bytes += chunk.len() as u64;
        if total_size > 0 {
            print!("\rDownloaded {:.2} / {:.2} MB ({:.2}%)", 
                  downloaded_bytes as f64 / 1_048_576.0, 
                  total_size as f64 / 1_048_576.0, 
                  (downloaded_bytes as f64 * 100.0) / total_size as f64);
        } else {
            print!("\rDownloaded {:.2} MB", downloaded_bytes as f64 / 1_048_576.0);
        }
        std::io::stdout().flush()?;
    }
    println!("\nDownload stream complete!");
    dest_file.flush().await?; // 确保所有数据都写入磁盘
    drop(dest_file);

    // 下载成功后，将临时文件重命名为目标文件
    tokio::fs::rename(&temp_file_path, file_path).await?;
    println!("Temporary file renamed to {}", file_path.display());

    Ok(all_bytes_for_session) // 返回下载的所有字节
}


impl ModelLoaderTrait for DefaultModelLoader {
    fn load_with_execution_providers(&self, model: Model, providers: Vec<ExecutionProviderDispatch>) -> Result<Session, Box<dyn Error>> {
        let root = env::current_dir()?;
        let model_root = root.join("models");
        if !model_root.exists() {
            fs::create_dir_all(&model_root)?;
        }
        match model {
            Model::Yolo11n => load_one_model(
                model_root.join("yolov11n_captcha.onnx").as_path(),
                &Url::parse("https://www.modelscope.cn/models/Amorter/CaptchaBreakerModels/resolve/master/yolov11n_captcha.onnx", )?,
                providers,
            ),
            Model::Siamese => load_one_model(
                model_root.join("siamese.onnx").as_path(),
                &Url::parse("https://www.modelscope.cn/models/Amorter/CaptchaBreakerModels/resolve/master/siamese.onnx", )?,
                providers,
            ),
        }
    }
}

impl ModelLoader {
    pub(crate) fn get_model_loader(self) -> Box<dyn ModelLoaderTrait> {
        match self {
            ModelLoader::DefaultModelLoader => Box::new(DefaultModelLoader::default()),
            ModelLoader::CustomModelLoader(model_loader) => model_loader,
        }
    }
}
