package db

import (
	"database/sql"
	"errors"
	"log"

	_ "modernc.org/sqlite"
)

// DB is the global database handle.
var DB *sql.DB

// InitDB opens the SQLite database and bootstraps the schema.
func InitDB(dbPath string) {
	var err error
	DB, err = sql.Open("sqlite", dbPath)
	if err != nil {
		log.Fatalf("[UCO Error] Failed to open SQLite database: %v", err)
	}

	// Enable Write-Ahead Logging (WAL) for concurrent read/write performance
	if _, err := DB.Exec("PRAGMA journal_mode=WAL;"); err != nil {
		log.Printf("[UCO Warning] Failed to set WAL journal mode: %v", err)
	}

	// 1. Create api_keys table
	_, err = DB.Exec(`
		CREATE TABLE IF NOT EXISTS api_keys (
			key_id TEXT PRIMARY KEY,
			client_name TEXT NOT NULL,
			active INTEGER NOT NULL DEFAULT 1,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		);
	`)
	if err != nil {
		log.Fatalf("[UCO Error] Failed to create api_keys table: %v", err)
	}

	// 2. Create request_metrics table
	_, err = DB.Exec(`
		CREATE TABLE IF NOT EXISTS request_metrics (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			model TEXT NOT NULL,
			original_text_tokens INTEGER NOT NULL,
			optimized_vision_tokens INTEGER NOT NULL,
			cost_savings_usd REAL NOT NULL
		);
	`)
	if err != nil {
		log.Fatalf("[UCO Error] Failed to create request_metrics table: %v", err)
	}

	// 3. Seed default API key if empty
	var count int
	err = DB.QueryRow("SELECT COUNT(*) FROM api_keys").Scan(&count)
	if err == nil && count == 0 {
		_, err = DB.Exec("INSERT INTO api_keys (key_id, client_name) VALUES ('uco-test-key-12345', 'UCO Default Test Client')")
		if err != nil {
			log.Printf("[UCO Warning] Failed to bootstrap default API key: %v", err)
		} else {
			log.Println("[UCO Info] Bootstrapped default API key: uco-test-key-12345")
		}
	}
}

// ValidateKey checks if the provided API key is valid and active in the database.
func ValidateKey(key string) (bool, string, error) {
	if DB == nil {
		return false, "", errors.New("database not initialized")
	}

	var clientName string
	var active int
	err := DB.QueryRow("SELECT client_name, active FROM api_keys WHERE key_id = ?", key).Scan(&clientName, &active)
	if err != nil {
		if err == sql.ErrNoRows {
			return false, "", nil // Key not found
		}
		return false, "", err
	}

	return active == 1, clientName, nil
}

// LogRequestMetric stores request execution stats into the database for analytical reporting.
func LogRequestMetric(model string, originalTextTokens, optimizedVisionTokens int, savingsUSD float64) error {
	if DB == nil {
		return errors.New("database not initialized")
	}

	_, err := DB.Exec(`
		INSERT INTO request_metrics (model, original_text_tokens, optimized_vision_tokens, cost_savings_usd)
		VALUES (?, ?, ?, ?)
	`, model, originalTextTokens, optimizedVisionTokens, savingsUSD)
	return err
}

// Ping checks the health of the SQLite database connection.
func Ping() error {
	if DB == nil {
		return errors.New("database connection is nil")
	}
	return DB.Ping()
}
