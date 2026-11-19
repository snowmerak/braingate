package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"os/signal"
	"strings"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

const usage = "usage: braingate -spec <file> [-context <text>] [-depth <n>] [-embed-url <url>] [-embed-model <name>] [-no-embed] [-seed <n>] <stimulus>"

type cliOptions struct {
	specPath        string
	stimulus        string
	previousContext string
	depth           int
	depthSet        bool
	embedURL        string
	embedModel      string
	noEmbed         bool
	seed            *int64
}

func main() {
	defaultEmbedURL := strings.TrimSpace(os.Getenv("BRAINGATE_EMBED_URL"))
	if defaultEmbedURL == "" {
		defaultEmbedURL = braingate.DefaultEmbedBaseURL
	}
	defaultEmbedModel := strings.TrimSpace(os.Getenv("BRAINGATE_EMBED_MODEL"))
	if defaultEmbedModel == "" {
		defaultEmbedModel = braingate.DefaultEmbedModel
	}

	specPath := flag.String("spec", "", "memory specification in JSON")
	previousContext := flag.String("context", "", "context preceding the stimulus")
	depth := flag.Int("depth", 0, "override temperament descent depth")
	embedURL := flag.String("embed-url", defaultEmbedURL, "OpenAI-compatible embeddings base URL")
	embedModel := flag.String("embed-model", defaultEmbedModel, "embedding model name")
	noEmbed := flag.Bool("no-embed", false, "disable embeddings and rank candidates by quadrant position only")
	seed := flag.Int64("seed", 0, "optional PRNG seed for deterministic pool sampling")
	flag.Usage = func() {
		_, _ = fmt.Fprintln(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()

	var depthSet, seedSet bool
	flag.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "depth":
			depthSet = true
		case "seed":
			seedSet = true
		}
	})

	opts := cliOptions{
		specPath:        *specPath,
		stimulus:        strings.Join(flag.Args(), " "),
		previousContext: *previousContext,
		depth:           *depth,
		depthSet:        depthSet,
		embedURL:        *embedURL,
		embedModel:      *embedModel,
		noEmbed:         *noEmbed,
	}
	if seedSet {
		s := *seed
		opts.seed = &s
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	err := run(ctx, opts, os.Stdout)
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, opts cliOptions, out io.Writer) error {
	embedFactory := func() (braingate.Embedder, error) {
		if opts.noEmbed {
			return nil, nil
		}
		return braingate.NewEmbedder(braingate.EmbedConfig{
			BaseURL: opts.embedURL,
			Model:   opts.embedModel,
		})
	}
	return runWith(ctx, newClientFromEnv, embedFactory, opts, out)
}

func runWith(
	ctx context.Context,
	getClient func() (*decisions.Client, error),
	getEmbedder func() (braingate.Embedder, error),
	opts cliOptions,
	out io.Writer,
) error {
	if strings.TrimSpace(opts.specPath) == "" || strings.TrimSpace(opts.stimulus) == "" {
		return errors.New(usage)
	}
	if opts.depth < 0 || (opts.depthSet && opts.depth == 0) {
		return errors.New("depth must be >= 1")
	}

	engine, err := loadSpec(opts.specPath)
	if err != nil {
		return err
	}
	if opts.depth > 0 {
		engine.Temperament.Depth = opts.depth
	}

	client, err := getClient()
	if err != nil {
		return err
	}

	var embedder braingate.Embedder
	if !opts.noEmbed && getEmbedder != nil {
		embedder, err = getEmbedder()
		if err != nil {
			return err
		}
	}

	var rng *rand.Rand
	if opts.seed != nil {
		rng = rand.New(rand.NewPCG(uint64(*opts.seed), 0))
	}

	query := strings.TrimSpace(opts.stimulus)
	if ctxText := strings.TrimSpace(opts.previousContext); ctxText != "" {
		query = ctxText + "\n" + query
	}

	input := struct {
		Stimulus string `json:"stimulus"`
		Context  string `json:"context,omitempty"`
	}{Stimulus: opts.stimulus, Context: opts.previousContext}

	result, err := engine.Run(ctx, client, decisions.Input{State: input}, braingate.RunOptions{
		Embedder: embedder,
		Query:    query,
		Rand:     rng,
	})
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

func loadSpec(path string) (*braingate.Engine, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	engine, err := braingate.ParseSpec(data)
	if err != nil {
		return nil, fmt.Errorf("spec %q: %w", path, err)
	}
	return engine, nil
}
