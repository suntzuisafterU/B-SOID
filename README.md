# B-SOiD

## Installation & Setup

1. Ensure that you have pip installed. You can confirm that is downloaded by running the following command: `pip --version`
2. TBD (TODO: High: Review: Run the following command: `pip install b-soid` (actual module name TBD))


## Usage

### Streamlit

To run normally: `streamlit run main.py streamlit`

To run the Streamlit app with an existing Pipeline file, run:

  - `streamlit run main.py streamlit -- -p '/path/to/existing.pipeline'`
    - This works with Linux and Windows systems so long as the path is absolute
    - Ensure that the path to the pipeline is in single quotes so that it is evaluated as raw (or else you 
    could have problems with backslashes and other weird characters)

### Jupyter Notebooks
- All you need to do is `import bsoid` and you're good to go! See the _notebooks_ folder for common usages.

  
------------------------------------------------------------------------------------------------------------------------

## Developer TODOs:

Things to refactor before extending functionality

### Build

- Originally: calling `bsoid_SUBMODULE.main.build(trainfolders)`
- Now: (TODO)


------------------------------------------------------------------------------------------------------------------------

#### Misc. Notes

Potential Abbreviations from legacy implementation
- sc:
  - scaled
  - scores
- fs:
  - frameshift



