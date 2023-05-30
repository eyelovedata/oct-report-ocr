# oct-report-ocr
OCR algorithms for OCT reports 

## extract_gcc_report_text_ou.py

- download gcc ou reports dicom files from the bucket
- extract text from gcc ou reports using OCR and the rule-based algo
- store the image of reports with their extracted text back to the bucket for future evaluation
- write text from reports to rows of csv file and store the csv file to the local directory
- delete the dicom file in the local directory after it got processed


## extract_onh_rnfl_report_text_ou.py

- download onh rnfl ou reports dicom files from the bucket
- extract text from onh rnfl ou reports using OCR and the rule-based algo
- store the image of reports with their extracted text back to the bucket for future evaluation
- write text from reports to rows of csv file and store the csv file to the local directory
- delete the dicom file in the local directory after it got processed


## upload_gcc_report_text_ou.py

- download gcc ou reports dicom files from the bucket
- extract text from gcc ou reports using OCR and the rule-based algo
- write text from reports to rows of csv file and store the csv file to the local directory
- delete the dicom file in the local directory after it got processed
- upload csv file to bigquery as a table


## upload_gcc_report_text_od.py

- download gcc od reports dicom files from the bucket
- extract text from gcc od reports using OCR and the rule-based algo
- write text from reports to rows of csv file and store the csv file to the local directory
- delete the dicom file in the local directory after it got processed
- upload csv file to bigquery as a table


## upload_gcc_report_text_os.py

- download gcc os reports dicom files from the bucket
- extract text from gcc os reports using OCR and the rule-based algo
- write text from reports to rows of csv file and store the csv file to the local directory
- delete the dicom file in the local directory after it got processed
- upload csv file to bigquery as a table


## upload_onh_rnfl_report_text_ou.py

- download onh rnfl ou reports dicom files from the bucket
- extract text from onh rnfl ou reports using OCR and the rule-based algo
- write text from reports to rows of csv file and store the csv file to the local directory
- delete the dicom file in the local directory after it got processed
- upload csv file to bigquery as a table


## upload_onh_rnfl_report_text_od.py

download onh rnfl od reports dicom files from the bucket
extract text from onh rnfl od reports using OCR and the rule-based algo
write text from reports to rows of csv file and store the csv file to the local directory
delete the dicom file in the local directory after it got processed
upload csv file to bigquery as a table


## upload_gcc_report_text_os.py

download onh rnfl os reports dicom files from the bucket
extract text from onh rnfl os reports using OCR and the rule-based algo
write text from reports to rows of csv file and store the csv file to the local directory
delete the dicom file in the local directory after it got processed
upload csv file to bigquery as a table


## utils.py

this file contains different functions and packages that help run the scripts above
