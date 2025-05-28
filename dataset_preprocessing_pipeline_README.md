# HAVEN: Hierarchical Attention for Viral protEin-based host iNference

This README contains information on how to use the dataset preprocessing pipeline of HAVEN.

## Dataset Preprocessing Pipeline
The [data_preprocessor.py](src/utils/scripts/data_preprocessor.py) allows you to process a dataset of viral protein sequences from [UniProt](https://www.uniprot.org/) or [UniRef](https://www.uniprot.org/uniref).


## Prerequisites

### Install [TaxonKit](https://bioinf.shenwei.me/taxonkit) and [pytaxonkit](https://github.com/bioforensics/pytaxonkit)
We used Taxonkit and PyTaxonkit to obtain the NCBI taxonomy information of viruses and virus hosts.
1. [Install taxonkit](https://bioinf.shenwei.me/taxonkit/download/#installation) using either Method 1 (preferred for Linux) or Method 2 (preferred for macOS)
2. [Download the taxonkit dataset](https://bioinf.shenwei.me/taxonkit/download/#dataset) and place it in the appropriate location. The absolute path to this directory must be provided using the `taxon_dir` argument where ever required in the below steps. 
3. [Install pytaxonkit](https://github.com/bioforensics/pytaxonkit).

> [!WARNING]
> The installation of taxonkit and pytaxonkit _may_ run into issues if you are using Windows OS. Linux and OSX are better suited. 

## Usage
The following functions are available in [data_preprocessor.py](src/utils/scripts/data_preprocessor.py):
1. Convert a uniprot or uniref fasta file into a csv file.
```shell
python .\src\utils\scripts\data_preprocessor.py --fasta_to_csv --input_file <absolute-path-to-fasta-file> --output_dir <absolute-path-to-the folder-where-the-output-file-will-be-written> --input_type <uniprot, uniref50, uniref90, or uniref100> --id_col <name-of-the-id-column-for-the-output-file-example: uniprot_id, uniref90_id, uniref50_id> 
```
2. Get metadata for each viral sequence from UniProt. The metadata includes host for the sequence as recorded in UniProt, and the EMBL id for the sequence. We use the EMBL id to query EMBL in the next step to get the virus host corresponding to each viral protein sequence.
```shell
python .\src\utils\scripts\data_preprocessor.py --uniprot_metadata --input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written> --input_type <uniprot, uniref50, uniref90, or uniref100 --id_col <name-of-the-id-column-defined-in-the-firs-step: uniprot_id, uniref90_id, uniref50_id> 
```
3. Query EMBL to get the virus hosts for each viral protein sequence using the EMBL id we fetched from UniProt in Step 2.
```shell
python .\src\utils\scripts\data_preprocessor.py --host_map_embl --input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written> --id_col <name-of-the-id-column-defined-in-the-firs-step: uniprot_id, uniref90_id, uniref50_id>
```
4. Remove viral protein sequences without a host from EMBL.
```shell
python .\src\utils\scripts\data_preprocessor.py --prune_dataset --input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written>
```
5. Get taxonomy information for the virus and virus hosts using taxonkit. 

> [!IMPORTANT]
> This step requires that taxonkit and pytaxonkit are installed. Refer [Installing taxonkit and pytaxonkit](#installing-taxonkit-and-pytaxonkit) for installation instructions.

```shell
python src/utils/scripts/data_preprocessor.py --taxon_metadata --input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written> --id_col <name-of-the-id-column-defined-in-the-firs-step: uniprot_id, uniref90_id, uniref50_id> --taxon_dir <absolute-path-to-the-taxon-directory-containing-taxonkit-files>
```
6. Filter for viruses at the species level
```shell
python src/utils/scripts/data_preprocessor.py --filter_species_virus --input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written>
```
7. Filter for virus hosts at the species level
```shell
python src/utils/scripts/data_preprocessor.py --filter_species_virus_host -input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written>
```
8. Filter for virus hosts belonging to the _Vertebrata_ clade (virus hosts that are vertebrates)
> [!IMPORTANT]
> This step requires that taxonkit and pytaxonkit are installed. Refer [Installing taxonkit and pytaxonkit](#installing-taxonkit-and-pytaxonkit) for installation instructions.

```shell
python src/utils/scripts/data_preprocessor.py --filter_vertebrates --input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written> --taxon_dir <absolute-path-to-the-taxon-directory-containing-taxonkit-files>
```
9. Merge the sequence data to create the final dataset file.
> [!NOTE]
> The output files in steps 2 to 8 do not contain the sequences, but only the sequence id (example 'uniport_id') and any metadata from uniprot or NCBI taxonomy.
> This was done intentionally to save memory and avoid creating multiple files with the entire sequence. 
> That is why we need the final step 9 to bring all the amino acid sequences back and create the final dataset file. You can use the `--merge_sequence_data` step at any point if you need one of the intermediate files along with the sequence data.
```shell
python src/utils/scripts/data_preprocessor.py --merge_sequence_data <absolute-path-to-the-output-file-from-step-1-that-contains-the-amino-acid-sequences> --input_file <absolute-path-to-the-input-file-(output-file-from-the-previous-step)> --output_dir <absolute-path-to-the-folder-where-the-output-file-will-be-written> --id_col <name-of-the-id-column-defined-in-the-firs-step: uniprot_id, uniref90_id, uniref50_id>
```

The output file from Step 9 would yield a dataset of all viral protein sequences with virus hosts extracted from EMBL, viruses and virus hosts at the species level, and virus hosts that are vertebrates.

This [jupyter notebook](src/jupyter_notebooks/datasets/generation/uniref90_embl_mapping_dataset_20240131_20240227.ipynb) contains code to do further custom filtering such as,
1. Select sequences of virus hosts with atleast one percent prevalence in the dataset.
2. Select sequences with length within the 99.9 percentile.
