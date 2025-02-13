## Cocoruta: A Legal Domain-Driven Q&A System

### Disclaimer
Cocoruta may reproduce biases and prejudices inherent in the legal documents used for its training, which include older legislation. Users should exercise caution when interpreting the model’s outputs, especially in contexts requiring up-to-date legal perspectives or that may involve underrepresented groups. We observed that *the Cocoruta model, while less proficient in handling utterances compared to larger models, would impart a legal bias to potential interactions.**

### Model Overview
[Cocoruta](https://huggingface.co/felipeoes/cocoruta-7b) is a specialized large language model fine-tuned for legal document-based Question Answering (Q&A), developed to address legal queries related to the "Blue Amazon"—a term used to describe Brazil's extensive maritime territory. Cocoruta 1.0 is based on the LLaMa 2-7B model, fine-tuned with a corpus of 68,991 legal documents totaling 28.4 million tokens. Despite being trained with fewer parameters than some larger models, Cocoruta demonstrates competitive performance in domain-specific legal discourse.

### Training and Technical Specifications
- **Parameter count**: 7B (LLaMa 2-7B)
- **Training data**: 28.4 million tokens from 68,991 legal documents
- **Training epochs**: 15

### Evaluation Metrics

#### Automatic Evaluation
Cocoruta has been evaluated using multiple automatic metrics to measure its effectiveness in generating accurate and relevant legal content. The model performed as follows:

- **BLEU**: 61.2
- **ROUGE-N**: 79.2
- **BERTSCORE**: 91.2
- **MOVERSCORE**: 76.5

#### Qualitative Evaluation
*The performance of Cocoruta in qualitative evaluation showed the utility of fine-tuning, as answers aligned with legal discourse were more frequent in Cocoruta compared to larger models. The larger models exhibited higher proficiency, delivering well-structured answers. However, for questions not directly related to the legal context, responses from the larger models did not maintain legal discourse.**
- **Adherence to legal discourse**: 74%
- **Correct answers**: 68%
- **Inappropriate discourse**: 51%

## Running the model within web interface

### Requirements
- Docker installed
- Windows or Linux OS

### Clone repo
```bash
git clone https://github.com/C4AI/cocoruta.git
```

### Change to repo directory
```bash
cd cocoruta
```

### Build Cocoruta UI docker image and run application
```bash
docker compose up
```

### Citation
If you use Cocoruta in your research, please cite the following paper:

```bibtex
*@inproceedings{2024cocoruta,
  author={do Espírito Santo, Felipe Oliveira and Marques Peres, Sarajane and de Sousa Gramacho, Givanildo and Alves Franco Brandão, Anarosa and Cozman, Fabio Gagliardi},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Legal Document-Based, Domain-Driven Q&A System: LLMs in Perspective}, 
  year={2024},
  volume={},
  number={},
  pages={1-9},
  keywords={Law;Large language models;Neural networks;Question answering (information retrieval);Complexity theory;Large language models;LLM evaluation;legal Q&A systems;legal-document corpus},
  address={Yokohama, Japan},
  isbn={978-8-3503-5931-2},
  doi={10.1109/IJCNN60899.2024.10650895},
  url={https://ieeexplore.ieee.org/abstract/document/10650895}
}
```

### Resources
- [Cocoruta Model Card](https://huggingface.co/felipeoes/cocoruta-7b)
- [Cocoruta Training and Evaluation Data](https://huggingface.co/datasets/felipeoes/cocoruta-evaluation)


### External Links
- [Cocoruta Paper](https://ieeexplore.ieee.org/abstract/document/10650895)
- [KEML Website](https://sites.usp.br/keml)
