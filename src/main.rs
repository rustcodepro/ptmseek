mod args;
mod burnecoder;
mod tchencoder;
mod tchlstm;
use crate::args::CommandParse;
use crate::args::Commands;
use crate::tchencoder::DNAencoder;
use clap::Parser;

/*
Gaurav Sablok
codeprog@icloud.com
*/

fn main() {
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::PTMEncoder {
            inputfastafile,
            inputdim,
            bottleneck,
            epochs,
            thread,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let filepath = DNAencoder {
                    pathfile: inputfastafile,
                };
                let command = filepath.run_encoder(inputfastafile, *inputdim, *bottleneck, *epochs);
                println!("The command has finished:{:?}", command)
            });
        }
        Commands::BurnEncoder {
            inputdim,
            bottlenext,
            epochs,
            thread,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let new = ();
            })
        }
        Commands::PTMLSTM {
            fastafile,
            layers,
            dropout,
            hiddenlayer,
            thread,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let function = ();
            })
        }
    }
}
