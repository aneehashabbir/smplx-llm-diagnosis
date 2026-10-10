# SMPL-X Body Measurement with LLM-Based Error Diagnosis

**Master's thesis project, Hohai University (Computer Science and Technology)**
*LLM-Enhanced Semantic Analysis for Diagnosing Measurement Errors in Monocular SMPL-X Body Reconstruction*

Author: Aneeha Shabbir · Supervisor: Prof. Wang Yu


**Interactive demo (prototype):** https://aneehashabbir.github.io/smplx-llm-diagnosis/demo.html
---

## What this project does

1. **Measures a body from one photo.** A 3D SMPL-X body is fitted to a single photograph with SMPLify-X, and shoulder width, chest, waist and hip are measured on it.
2. **Checks the measurements** against tape measurements (own participants) or true 3D body scans (public HBW benchmark).
3. **Explains each error with a vision-language model.** For every measurement that is more than 10% off, a multimodal LLM looks at the photo and explains *why* it is wrong and *how* to fix it. Simple code rules decide the facts (tolerance, error direction, body size) and reject physically impossible answers.

```
photo → keypoints → SMPLify-X (SMPL-X body) → measure_body.py → errors
                                                                  ↓
                         per-person report  ←  LLM diagnosis (checked by code rules)
```

## Main findings (October 2026, work in progress)

- **Measurement accuracy.** Own participants (6 people, tape measure): mean error shoulder 8.3%, chest 7.0%, waist 7.6%, hip 6.7%. HBW benchmark (80 photos of 10 people, 3D scans): shoulder 9.5%, chest 10.7%, waist 11.6%, hip 6.2%.
- **Three causes of error:** the body is built at the **wrong overall size** (one photo cannot tell how tall a person is), **missing depth** in a single image, and **loose clothing**.
- **Clothing matters more in everyday photos:** for the same HBW people, chest and waist error rise from about 8–9% (lab photos) to 13–14% (everyday photos).
- **Errors cancel out on average.** Because the causes push measurements in opposite directions, the average error can look small while individual errors are large (HBW waist: average −0.9 cm, typical individual error about 10 cm). This is why a per-person diagnosis is needed.
- **LLM diagnosis on HBW** (GLM-5.3-Flash, 121 wrong measurements): body size 59, depth 22, clothing 22 (17 of them in everyday photos), pose 13. Every "body size" answer matched the real height error, and the code rules allowed no physically impossible answers.

## Files

| File | What it does |
| --- | --- |
| `measure_body.py` | Measures SMPL-X bodies the standard (SHAPY/HBW) way: standard pose, chest/waist/hip at body landmarks, shoulder over the back, widest hip. Works for own participants and for HBW. Never overwrites existing results. |
| `llm_diagnosis_new.py` | LLM diagnosis for one person: error categories, tolerance and sign rules decided by code, size check, plain-English report. |
| `run_all_new.py` | Runs the diagnosis for all people (own data or HBW), with GLM and DeepSeek, and builds comparison tables and a report. |
| `select_hbw_photos.py` | Picks clear, full-body, front-facing photos from the HBW validation set and prepares them for SMPLify-X. |
| `compare_scripts_hbw.py` | Compares the first measuring script with `measure_body.py` on HBW. |
| `shoulder_check.py` | Compares four ways of measuring shoulder width on one body. |
| `extract_measurements.py` | First version of the measuring script (kept for comparison). |
| `examples/` | Example formats for participant metadata and tape measurements (no real data). |

## Setup

1. **Python 3.11** and the packages: `pip install -r requirements.txt`
2. **SMPL-X body models:** register at https://smpl-x.is.tue.mpg.de and put the model files in `models/smplx/` (`SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`, `SMPLX_NEUTRAL.npz`).
3. **SMPLify-X:** follow https://github.com/vchoutas/smplify-x, including the VPoser model (`V02_05`).
4. **HBW benchmark (optional):** register at https://shapy.is.tue.mpg.de and download the HBW validation set.
5. **API keys** are read from environment variables, never from the code:
   ```
   setx ZHIPUAI_API_KEY "your_key"
   setx DEEPSEEK_API_KEY "your_key"
   ```

## Usage

**Own participants** (one photo per person; tape values in `ground_truth/measurements.csv`, see `examples/`):
```
# 1. fit the 3D body with SMPLify-X (see the SMPLify-X documentation)
# 2. measure it
python measure_body.py --pkl results\P01_Front\000.pkl --gender male --subject_id P01 --gt ground_truth\measurements.csv --out shape_results\results_P01_shape.json
# 3. diagnose with the LLM
python run_all_new.py --dataset own --repeats 3 --glm-model glm-5.3-flash --thinking enabled --out-dir diagnoses_own
```

**HBW benchmark:**
```
python select_hbw_photos.py --hbw D:\HBW\HBW --out hbw_selected
# fit all selected photos with SMPLify-X, then:
python measure_body.py --hbw_gt_folder D:\HBW\HBW\smplx\val --gender neutral --out hbw_gt_measurements.json
python measure_body.py --hbw_pred_folder results --hbw_gt hbw_gt_measurements.json --gender neutral --out hbw_pred_measurements.json
python run_all_new.py --dataset hbw --glm-model glm-5.3-flash --thinking enabled --out-dir diagnoses_hbw
```

## Data and privacy

Photos, measurements and metadata of participants are **not** included, to protect their privacy. SMPL-X models, VPoser and the HBW dataset are **not** included because their licences do not allow redistribution; please download them from the original sources.

## Acknowledgements

This project builds on SMPL-X and SMPLify-X (Pavlakos et al., CVPR 2019), MediaPipe, the HBW benchmark and measurement definitions from SHAPY (Choutas et al., CVPR 2022), and the GLM (Zhipu AI) and DeepSeek language models.
