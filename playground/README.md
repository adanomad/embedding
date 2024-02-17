# Python Script for Processing Text Files with ChatGPT

This Python script is designed to read a text file and a prompt file, process the text file by splitting it by page numbers, and then send each page along with the prompt to a ChatGPT API. The script handles these operations in parallel, using up to 8 workers for efficiency.

# Features

- File Reading: Reads a specified text file and splits it by page numbers.
- Prompt Processing: Reads a prompt file and replaces a placeholder with the actual page text.
- ChatGPT Integration: Sends processed data to a ChatGPT API and retrieves responses.
- Parallel Processing: Handles multiple pages concurrently, improving processing speed.
- JSON Validation: Validates the JSON response from the API.
- Response Handling: Prints the JSON response (can be modified to store the response).

# Example

```
python3 process.py --txtfile DAVEBUSTER\'SENTERTAINMENTINC_20220629_8-K_EX-101_CreditLoanAgreement.PDF.txt --prompt page_data.prompt.txt
```

The output of the ChatGPT API should be something like:

```json
{
  "page_number": 1,
  "tranche": null,
  "quantum": null,
  "financial_maintenance_covenant": null,
  "addbacks_cap": null,
  "MFN_threshold": null,
  "MFN_exceptions": null,
  "portability": null,
  "lender_counsel": null,
  "borrower_counsel": null,
  "borrower": "DAVE & BUSTER’S, INC., as the Borrower",
  "guarantor": "THE OTHER GUARANTORS PARTY HERETO FROM TIME TO TIME",
  "admin_agent": "DEUTSCHE BANK AG NEW YORK BRANCH, as Administrative Agent and Collateral Agent",
  "collat_agent": "DEUTSCHE BANK AG NEW YORK BRANCH, as Collateral Agent",
  "effective_date": "June 29, 2022"
}
```
