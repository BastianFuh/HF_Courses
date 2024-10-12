"""Main file for executing the course scripts.
This file exists so that some QoL functionality can be exported an be used across all tasks. However
this requires the course files to be handled as part of a package and, therefore, can't easily be called
themself directly.
"""

ACTIVE_CHAPTER = 6

ACTIVE_SECTIONS = [8]

if __name__ == "__main__":
    # NLP Course
    # Chapter 1
    if ACTIVE_CHAPTER == 1:
        if 1 in ACTIVE_SECTIONS:
            import HF_NLP_Course._01_Transformer_Models._01_Basic

    # Chapter 2
    if ACTIVE_CHAPTER == 2:
        if 1 in ACTIVE_SECTIONS:
            import HF_NLP_Course._02_Using_Transformers._01_Basics

        if 2 in ACTIVE_SECTIONS:
            import HF_NLP_Course._02_Using_Transformers._02_Models

        if 3 in ACTIVE_SECTIONS:
            import HF_NLP_Course._02_Using_Transformers._03_Tokenizers

        if 4 in ACTIVE_SECTIONS:
            import HF_NLP_Course._02_Using_Transformers._04_Sequences

        if 5 in ACTIVE_SECTIONS:
            import HF_NLP_Course._02_Using_Transformers._05_PuttingItTogether

    # Chapter 3
    if ACTIVE_CHAPTER == 3:
        if 1 in ACTIVE_SECTIONS:
            import HF_NLP_Course._03_Fine_Tuning._01_ProcessingTheData

        if 2 in ACTIVE_SECTIONS:
            import HF_NLP_Course._03_Fine_Tuning._02_Training

        if 3 in ACTIVE_SECTIONS:
            import HF_NLP_Course._03_Fine_Tuning._03_FullTraining

    # Chapter 4
    # This chapter focused on explaining the process of uploading and sharing modles
    # on the hub. So there was nothing to programm
    if ACTIVE_CHAPTER == 4:
        pass

    # Chapter 5
    if ACTIVE_CHAPTER == 5:
        if 1 in ACTIVE_SECTIONS:
            import HF_NLP_Course._05_DatasetLibrary._01_ExternalDatasets

        if 2 in ACTIVE_SECTIONS:
            import HF_NLP_Course._05_DatasetLibrary._02_DataManipulation

        if 3 in ACTIVE_SECTIONS:
            import HF_NLP_Course._05_DatasetLibrary._03_BigData

        if 4 in ACTIVE_SECTIONS:
            import HF_NLP_Course._05_DatasetLibrary._04_CreatingADataset

        if 5 in ACTIVE_SECTIONS:
            import HF_NLP_Course._05_DatasetLibrary._05_SemanticSearchFAISS

    # Chapter 6
    if ACTIVE_CHAPTER == 6:
        if 1 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._01_TrainingATokenizer

        if 2 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._02_FastTokenizerFeatures

        if 3 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._03_QA_Pipeline

        if 4 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._04_NormalizationPreTokenization

        if 5 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._05_BytePairEncoding

        if 6 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._06_WordPieceTokenization

        if 7 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._07_UnigramTokenization

        if 8 in ACTIVE_SECTIONS:
            import HF_NLP_Course._06_Tokenizer._08_BuildingTokenizer

    # Chapter 7
    if ACTIVE_CHAPTER == 7:
        pass

    # Chapter 8
    if ACTIVE_CHAPTER == 8:
        pass

    # Chapter 9
    if ACTIVE_CHAPTER == 9:
        pass
