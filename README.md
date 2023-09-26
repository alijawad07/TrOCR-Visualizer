# TrOCR-Visualizer

A demonstration of TrOCR's capabilities in Optical Character Recognition (OCR), with the results compiled into a video. This repository showcases how TrOCR can be used for accurate text extraction from both printed and handwritten samples.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone this repository and install the required packages:

```bash
git clone https://github.com/alijawad07/TrOCR-Visualizer.git
cd TrOCR-Visualizer
pip install -r requirements.txt
```

## Usage

To run the OCR evaluation, execute:

```bash
python main.py --data_path=<path_to_images> --num_samples=<number_of_samples>
```

Replace `<path_to_images>` with the directory path containing the images you want to process, and `<number_of_samples>` with the number of samples you want to evaluate.

## Features

- Utilizes TrOCR for OCR tasks.
- Includes a feature to save OCR results into a dynamic-resolution video.
- Works out-of-the-box with high accuracy on printed text.

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

[MIT](LICENSE)
