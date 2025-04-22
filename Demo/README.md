## Demo Script For Reliable, Explainable Brain Tumor Segmentation

### Folder Structure

```
Demo/
├── BestModel/
│   ├── model.pth
│   └── model.json
├── Data/
├── demo.py
├── requirements.txt
├── environment.yml
└── README.md
```

### Steps
1. Create an environment
   ```sh
    conda env create -f environment.yml
    conda activate AuditpulseBackend
    pip install -r requirements.txt
   ```
2. Activate the environment
   ```sh
    conda activate AuditpulseBackend
    pip install -r requirements.txt
   ```
3. Install dependencies
   ```sh
    pip install -r requirements.txt
   ```
4. Run
   ```sh
    python demo.py
   ```
**Note** 
- GPU is needed to run the inference in reasonable time.
- If depencies missing, please install using `pip install module_name`.