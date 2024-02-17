package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
)

type ChatGPTResponse struct {
	OriginalPageText             string  `json:"original_page_text"`
	PageNumber                   int     `json:"page_number"`
	Summary                      string  `json:"summary"`
	Tranche                      *string `json:"tranche"`
	Quantum                      *string `json:"quantum"`
	FinancialMaintenanceCovenant *string `json:"financial_maintenance_covenant"`
	AddbacksCap                  *string `json:"addbacks_cap"`
	MFNThreshold                 *string `json:"MFN_threshold"`
	MFNExceptions                *string `json:"MFN_exceptions"`
	Portability                  *string `json:"portability"`
	LenderCounsel                *string `json:"lender_counsel"`
	BorrowerCounsel              *string `json:"borrower_counsel"`
	Borrower                     *string `json:"borrower"`
	Guarantor                    *string `json:"guarantor"`
	AdminAgent                   *string `json:"admin_agent"`
	CollatAgent                  *string `json:"collat_agent"`
	EffectiveDate                *string `json:"effective_date"`
}

func main() {
	txtFile := flag.String("txtfile", "DAVEBUSTER.txt", "Path to the text file")
	promptFile := flag.String("prompt", "credit.prompt.txt", "Path to the prompt file")
	csvFile := flag.String("csv", "DAVEBUSTER.csv", "Path to the CSV file")
	flag.Parse()

	pages := readAndSplitFile(*txtFile)
	fmt.Printf("Read %d pages from %s\n", len(pages), *txtFile)

	var wg sync.WaitGroup
	responses := make([]ChatGPTResponse, len(pages))
	for i, page := range pages {
		wg.Add(1)
		go func(idx int, pg string) {
			defer wg.Done()
			responses[idx] = sendToChatGPT(pg, *promptFile)
		}(i, page)
	}
	wg.Wait()

	exportToCSV(responses, *csvFile)
	fmt.Println("Exported to CSV:", *csvFile)
}

func readAndSplitFile(filePath string) []string {
	file, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var pages []string
	scanner := bufio.NewScanner(file)
	var currentPage strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "<PAGE ") {
			pages = append(pages, currentPage.String())
			currentPage.Reset()
			continue
		}
		currentPage.WriteString(line + " ")
	}
	pages = append(pages, currentPage.String())

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	return pages
}

func sendToChatGPT(pageText, promptPath string) ChatGPTResponse {
	prompt, err := os.ReadFile(promptPath)
	if err != nil {
		panic(err)
	}

	// Prepare the request payload
	requestData := map[string]interface{}{
		"model":      "gpt-4-1106-preview",
		"prompt":     strings.Replace(string(prompt), "[page_text]", pageText, -1),
		"max_tokens": 8096,
	}

	requestDataJSON, err := json.Marshal(requestData)
	if err != nil {
		panic(err)
	}

	// Set up HTTP request
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/engines/davinci-codex/completions", strings.NewReader(string(requestDataJSON)))
	if err != nil {
		panic(err)
	}

	// Set the necessary headers, including the OpenAI API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		panic("No OpenAI API key found in environment variables")
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// Perform the HTTP request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	// Read and unmarshal the response
	responseBody, err := io.ReadAll(io.Reader(resp.Body))
	if err != nil {
		panic(err)
	}

	var response ChatGPTResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		panic(err)
	}

	return response
}

func exportToCSV(data []ChatGPTResponse, filePath string) {
	file, err := os.Create(filePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write CSV headers
	headers := []string{"Page Number", "Summary", "Original Page Text", "Tranche"}
	if err := writer.Write(headers); err != nil {
		panic(err)
	}

	// Write data to CSV
	for _, d := range data {
		var record []string
		record = append(record, fmt.Sprintf("%d", d.PageNumber), d.Summary, d.OriginalPageText)
		if d.Tranche != nil {
			record = append(record, *d.Tranche)
		} else {
			record = append(record, "")
		}
		if err := writer.Write(record); err != nil {
			panic(err)
		}
	}
}
