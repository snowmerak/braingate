package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

const usage = "usage: braingate -spec <file> [-context <text>] <stimulus>"

func main() {
	specPath := flag.String("spec", "", "gate specification in JSON")
	previousContext := flag.String("context", "", "context preceding the stimulus")
	flag.Usage = func() {
		_, _ = fmt.Fprintln(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	err := run(ctx, *specPath, strings.Join(flag.Args(), " "), *previousContext, os.Stdout)
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, specPath, stimulus, previousContext string, out io.Writer) error {
	return runWith(ctx, newClientFromEnv, specPath, stimulus, previousContext, out)
}

func runWith(ctx context.Context, getClient func() (*decisions.Client, error), specPath, stimulus, previousContext string, out io.Writer) error {
	if strings.TrimSpace(specPath) == "" || strings.TrimSpace(stimulus) == "" {
		return errors.New(usage)
	}
	graph, err := loadSpec(specPath)
	if err != nil {
		return err
	}
	client, err := getClient()
	if err != nil {
		return err
	}

	input := struct {
		Stimulus string `json:"stimulus"`
		Context  string `json:"context,omitempty"`
	}{Stimulus: stimulus, Context: previousContext}

	result, err := graph.Run(ctx, client, decisions.Input{State: input})
	if err != nil {
		return err
	}

	encoder := json.NewEncoder(out)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}

func newClientFromEnv() (*decisions.Client, error) {
	apiKey := strings.TrimSpace(os.Getenv("TYPESAFE_API_KEY"))
	if apiKey == "" {
		return nil, errors.New("TYPESAFE_API_KEY is required")
	}
	return decisions.NewClient(decisions.Config{
		API:    decisions.SystemOneAPI,
		Model:  "jev-latest",
		APIKey: apiKey,
	})
}

func loadSpec(path string) (*braingate.Graph, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	graph, err := braingate.ParseSpec(data)
	if err != nil {
		return nil, fmt.Errorf("spec %q: %w", path, err)
	}
	return graph, nil
}
