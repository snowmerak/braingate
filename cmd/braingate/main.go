package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/braingate/internal/incidentdemo"
	"github.com/snowmerak/decisions"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	err := run(ctx, os.Args[1:])
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string) error {
	apiKey := strings.TrimSpace(os.Getenv("TYPESAFE_API_KEY"))
	if apiKey == "" {
		return errors.New("TYPESAFE_API_KEY is required")
	}
	client, err := decisions.NewClient(decisions.Config{
		API:    decisions.SystemOneAPI,
		Model:  "jev-latest",
		APIKey: apiKey,
	})
	if err != nil {
		return err
	}

	input := strings.TrimSpace(strings.Join(args, " "))
	if input == "" {
		input = incidentdemo.SampleInput
	}
	history, err := incidentdemo.NewGraph().Run(ctx, client, decisions.Input{State: input})
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(os.Stdout, braingate.ComposeText(history))
	return err
}
