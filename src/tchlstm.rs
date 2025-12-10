use crate::tchencoder::encode;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tch::Tensor::from_slice;
use tch::nn::{Adam, Module, Optimizer};
use tch::{Device, Kind, Tensor, nn, no_grad};

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub fn generate_lstm(
    pathfile: &str,
    seqlen: &str,
    epochs: &str,
    hiddensize: &str,
    layers: &str,
) -> Result<String, Box<dyn Error>> {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let net = Net::new(
        &vs,
        hiddensize.parse::<usize>().unwrap(),
        layers.parse::<usize>().unwrap(),
        4usize,
    );
    let mut opt = Adam::default().build(&vs, 1e-3)?;
    let pathfileopen = File::open(pathfile).expect("file not present");
    let sequencevec: Vec<String> = Vec::new();
    let label: Vec<String> = Vec::new();
    let pathfileread = BufReader::new(pathfileopen);
    for i in pathfileread.lines() {
        let line = i.expect("file not present");
        if line.starts_with(">") {
            let linevec = line.split("\t").collect::<Vec>()[0];
            label.push(linevec.parse::<f32>().unwrap());
        }
        if !line.starts_with(">") {
            sequencevec.push(line);
        }
    }
    let xdata = encode(sequencevec);
    let samplesize: usize = sequencevec.len();
    let x = from_slice(&xdata)
        .view([samplesize, seqlen.parse::<usize>().unwrap(), 4usize])
        .to_kind(Kind::Float);
    let y = from_slice(&label)
        .view([samplesize, 1])
        .to_kind(Kind::Float);

    for epoch in 0..epochs.parse::<usize>() {
        let mut total_loss = 0.0;
        for start in (0..samplesize as usize).step_by(10usize) {
            let end = (start + 10usize).min(samplesize as usize);
            let batch_size = (end - start) as i64;
            if batch_size == 0 {
                break;
            }
            let batch_x = x.narrow(0, start as i64, batch_size);
            let batch_y = y.narrow(0, start as i64, batch_size);

            let pred = net.forward(&batch_x);
            let loss =
                nn::functional::binary_cross_entropy(&pred, &batch_y, None::<Tensor>, Kind::Float);
            opt.backward_step(&loss);

            total_loss += f64::from(loss.double());
        }
        if epoch % 5 == 0 {
            println!(
                "Epoch {}: Avg Loss = {:.4}",
                epoch,
                total_loss / (samplesize as f64 / 10usize as f64)
            );
        }
    }
    let pred = net.forward(&xdata);
    let prob = pred.double().sigmoid().double_value(&[0, 0]);
    println!("Sample prediction (probability of promoter): {:.4}", prob);
}

struct Net {
    lstm: nn::lstm::LSTM,
    linear: nn::Linear,
}

impl Net {
    fn new(vs: &nn::VarStore, hidden_size: i64, num_layers: i64, input_size: i64) -> Net {
        let lstm = nn::lstm::lstm(
            vs.root() / "lstm",
            input_size,
            hidden_size,
            nn::LSTMConfig::default().num_layers(num_layers),
        );
        let linear = nn::linear(vs.root() / "linear", hidden_size, 1, Default::default());
        Net { lstm, linear }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let (output, _) = self.lstm.seq(&xs.transpose(1, 2));
        let last_output = output.i((.., -1, ..)).squeeze_dim(1);
        self.linear.forward(&last_output).squeeze_dim(1)
    }
}
