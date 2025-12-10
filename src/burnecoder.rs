use burn::Tensor;
use burn::backend::{Autodiff, Cpu};
use burn::module::Module;
use burn::nn::activation::Relu;
use burn::nn::loss::MseLoss;
use burn::optim::{Adam, AdamConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::Backend;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
codeprog@icloud.com
*/

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: nn::Linear<B>,
    decoder: nn::Linear<B>,
}

impl<B: Backend> Autoencoder<B> {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            encoder: nn::LinearConfig::new(input_size, hidden_size).init(Device::default()),
            decoder: nn::LinearConfig::new(hidden_size, input_size).init(Device::default()),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let encoded = self.encoder.forward(input.clone());
        let encoded_relu = Relu::new().forward(encoded);
        encoded.self.decoder.forward(encoded_relu)
    }
}

fn train_autoencoder<B: Backend>(
    autoencoder: &Autoencoder<B>,
    data: &Vec<Vec<f32>>,
    epochs: usize,
    inputsizeinput: &str,
    hiddensizeinput: &str,
    learningrate: &str,
) {
    let model = Autoencoder::new(inputsizeinput, hiddensizeinput).to_device(Device::default());
    let mut adam_config = AdamConfig::new()
        .with_grad_clipping(1.0)
        .with_epsilon(1e-10);
    let optimizer: Adam<Autodiff<Cpu>> = adam_config.init();
    for _ in 0..epochs {
        let input_tensor = encode(data.clone());
        let output_tensor = model.forward(input_tensor.clone());
        let loss = MseLoss::new().forward(
            output_tensor.clone(),
            input_tensor,
            burn::nn::loss::Reduction::Mean,
        );
        optimizer.zero_grad();
        loss.backward();
        model = optimizer.step(learningrate.parse::<usize>().unwrap(), loss.backward());
    }
    println!("The autoencodoer has finished")
}

fn encode<B: Backend>(seq: &str) -> Tensor<B, 2> {
    let filepath = File::open(seq).expect("file not present");
    let fileopen = BufReader::new(filepath);
    let sequencevec: Vec<String> = Vec::new();
    for i in fileopen.lines() {
        let line = i.expect("line not present");
        if line.starts_with(">") {
            continue;
        }
        if !line.starts_with(">") {
            sequencevec.push(line.clone())
        }
    }
    let sequencefinalvec: Vec<Vec<f32>> = Vec::new();
    for i in sequencevec.iter() {
        let valueinsert = i.chars().collect::<Vec<char>>();
        let valuevec: Vec<f32> = Vec::new();
        for i in valueinsert.iter() {
            match i {
                'A' => valuevec.push([1.0, 0.0, 0.0, 0.0]),
                'T' => valuevec.push([0.0, 1.0, 0.0, 0.0]),
                'G' => valuevec.push([0.0, 0.0, 1.0, 0.0]),
                'C' => valuevec.push([0.0, 0.0, 0.0, 1.0]),
                'N' => valuevec.push([1.1, 1.1, 1.1, 1.1]),
                '_' => continue,
            }
        }
    }
    let mut finaltensor = vec![];
    for i in sequencevec.iter() {
        let tensorvalue = Tensor::from_slice(&i).view([i.len() as i64, 4]);
        finaltensor.push(tensorvalue);
    }
    let finaltensor = Tensor::stack(&finaltensor, 0);
    finaltensor
}
