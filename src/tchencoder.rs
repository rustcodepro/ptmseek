use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::vec;
use tch::{Device, Tensor, nn, nn::Module, nn::OptimizerConfig};

/*
Gaurav Sablok
codeprog@icloud.com
*/

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct DNAencoder {
    pub pathfile: String,
}

#[derive(Debug)]
pub struct Autoencoder {
    encoder: nn::Sequential,
    decoder: nn::Sequential,
}

impl Autoencoder {
    fn toxencoder(vs: &nn::Path, input_dim: i64, bottleneck_dim: i64) -> Self {
        let encoder = nn::seq()
            .add(nn::linear(vs / "enc1", input_dim, 128, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(
                vs / "enc2",
                128,
                bottleneck_dim,
                Default::default(),
            ));
        let decoder = nn::seq()
            .add(nn::linear(
                vs / "dec1",
                bottleneck_dim,
                128,
                Default::default(),
            ))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(vs / "dec2", 128, input_dim, Default::default()));
        Self { encoder, decoder }
    }
}

impl Module for Autoencoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let encoded = self.encoder.forward(xs);
        self.decoder.forward(&encoded)
    }
}

impl DNAencoder {
    pub fn run_encoder(
        &self,
        fastafile: &str,
        inputdim: i64,
        bottleneck: i64,
        epochs: i64,
    ) -> Result<String, Box<dyn Error>> {
        let traindata = encode(fastafile);
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let autoencoder = Autoencoder::toxencoder(&vs.root(), inputdim, bottleneck);
        let learning_rates = [0.001, 0.0001, 0.01];
        let filenames = [
            "encoded.model_0.001",
            "encoded.model_0.0001",
            "encoded.model_0.01",
        ];
        let mse_loss = |pred: &Tensor, target: &Tensor| pred.mse_loss(target, tch::Reduction::Mean);
        for (lr, _filename) in learning_rates.iter().zip(filenames.iter()) {
            println!("Training with learning rate: {}", lr);
            let mut opt = nn::Adam::default().build(&vs, *lr).unwrap();
            for epoch in 1..=epochs {
                opt.zero_grad();
                let output = autoencoder.forward(&traindata);
                let loss = mse_loss(&output, &traindata);
                let lossvalue = loss.to_kind(tch::Kind::Float).double_value(&[]);
                loss.backward();
                opt.step();
                if epoch % 10 == 0 || epoch == 1 {
                    println!("Epoch [{}/{}/{}]", epoch, epochs, lossvalue);
                }
            }
        }
        Ok("The autoencoder has finished".to_string())
    }
}

/*
 A multifunctional tensor from the multifasta file.Stacking all the tensors
 All the sequences should be of the same length.
*/

pub fn encode(pathfile: &str) -> Tensor {
    let filepath = File::open(pathfile).expect("file not present");
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
