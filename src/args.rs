use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "toxbert",
    version = "1.0",
    about = "PTM autoencoder and LSTM classifier.
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// prepare the encoder for the DNA sequences
    PTMEncoder {
        /// input file fasta
        inputfastafile: String,
        /// inputdimension
        inputdim: i64,
        ///bottleneck dimension
        bottleneck: i64,
        /// epochs
        epochs: i64,
        /// threads for the analysis
        thread: String,
    },
    /// implement the classsification model on the PTM sequences
    PTMLSTM {
        /// input fasta file
        fastafile: String,
        /// num of layers
        layers: String,
        /// dropout layer
        dropout: String,
        /// hidden layer for the analysis
        hiddenlayer: String,
        /// threads for the analysis
        thread: String,
    },
    /// Autoencoder using the tch crate
    BurnEncoder {
        /// input dimension
        inputdim: i64,
        /// bottleneck dimension
        bottlenext: i64,
        /// epochs
        epochs: i64,
        /// threads for the analysis
        thread: String,
    },
}
