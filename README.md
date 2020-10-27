
To run streamlit, run: `streamlit run main.py streamlit`




---

---

Formal readme below

# B-SOiD

## Installation

1. Ensure that you have pip installed
2. (Run the following command: `pip install b-soid` (actual module name TBD)) // TODO: med: review



------------------------------------------------------------------------------------------------------------------------

## Setup

Setup instructions go here

TODO

------------------------------------------------------------------------------------------------------------------------

## Usage

TODO: med: Usage instructions go here

### Streamlit

To run normally: `streamlit run main.py streamlit`

To run the Streamlit app with an existing Pipeline file, run:

  - `streamlit run main.py streamlit -- -p '/path/to/existing.pipeline'`
    - This works with Linux and Windows systems so long as the path is absolute
    - Ensure that the path to the pipeline is in single quotes so that it is evaluated as raw (or else you 
    could have problems with backslashes and other weird characters)





------------------------------------------------------------------------------------------------------------------------

#### Misc. Notes

Potential Abbreviations from legacy implementation
- sc:
  - scaled
  - scores
- fs:
  - frameshift
  
------------------------------------------------------------------------------------------------------------------------

## Developer TODOs:

Things to refactor before extending functionality

### Build

- Originally: calling `bsoid_SUBMODULE.main.build(trainfolders)`
- Now: (TODO)