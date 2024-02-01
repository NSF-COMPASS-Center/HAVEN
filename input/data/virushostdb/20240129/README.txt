Virus-Host DB: Database of sequenced viruses and their hosts
====================================================

Laboratory of Chemical Life Science, Bioinformatics Center,
Institute for Chemical Research, Kyoto University
Uji, Kyoto 611-0011, Japan
Web site: http://www.genome.jp/virushostdb/
Feedback: http://www.genome.jp/feedback/
==================================================================

Virus-Host DB organizes data about the relationships between viruses and their
hosts, represented in the form of pairs of NCBI taxonomy IDs (corresponding to the 
NCBI taxonomy available when the Virus-Host DB is updated) for viruses and
their hosts. Virus-Host DB covers viruses with complete genomes stored in
1) NCBI/RefSeq and 2) GenBank whose accession numbers are listed in EBI Genomes.
The host information is collected from RefSeq, GenBank (in free text format),
UniProt, ViralZone, and manually curated with additional information obtained
by literature surveys.

This directory contains following files.

virushostdb.tsv: Tab separated file containing the following information:
                 virus tax id    ... tax ID of a virus
                 virus name      ... name of a virus
                 virus lineage   ... lineage of a virus
                 refseq id       ... RefSeq IDs of a virus
                 KEGG GENOME     ... Dblink to KEGG GENOME
                 KEGG DISEASE    ... Dblink to KEGG DISEASE
                 DISEASE         ... disease name
                 host tax id     ... tax ID of a host
                 host name       ... name of a host
                 host lineage    ... lineage of a host
                 pmid            ... PubMed ID
                 evidence        ... source of the host information
                 sample type     ... the type of sample from which the environmental viral genome was acquired (host tax id is 1 in this case)
                 source organism ... if the sample type specifies a group of organisms (e.g. Animal or Plant), the tax ID of the source organism is provided here.

virushostdb.gbff.gz: GenBank formatted genome information of the viruses.
virushostdb.formatted.genomic.fna.gz: FASTA formatted sequences of their genomes. Sequence name is "|" separated and contains:
		Sequence_accession virus name
		Hostname
		Virus lineage
		Host lineage
		Sample type (if the viral genome is derived from metagenome)
		Taxonomic identifier (if sample type is "Organismal")

virushostdb.formatted.cds.faa.gz: FASTA formatted protein sequences of their coding regions. Sequence name is "|" separated and contains:
		Sequence_accession virus name
		gene name
                Hostname
                Virus lineage 
                Host lineage
                Sample type
                Taxonomic identifier (if sample type is "Organismal")

virushostdb.formatted.cds.fna.gz: FASTA formatted nucleotide sequences of their coding regions (format is same as above).
taxid2parents_VH.tsv: list of parent taxa for each taxon
taxid2lineage_full_VH.tsv: full lineage for each taxon
taxid2lineage_abbreviated_VH.tsv: abbreviated lineage for each taxon

non-segmented_virus_list.tsv: Tab separated file listing name, NCBI taxonomic identifier and Sequence accession for viruses having a non-segmented genome. 
segmented_virus_list.tsv: as above for viruses having a segmented genome.

virushostdb_comment.daily.tsv: comment data

==================================================================
Change Log
[2018/08/08] added sample type, source organism columns to virushostdb.tsv.
[2018/12/16] added description of fasta header.
[2019/05/30] added lists of viruses having segmented and non-segmented genome.
[2021/10/05] added virushostdb_comment.daily.tsv

==================================================================
Last update: 2021/10/05