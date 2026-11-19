package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

const usage = "usage: braingate -spec <file> [-persona <text>] [-context <text>] [-depth <n>] [-pool-size <n>] [-min-p <ratio>] [-now <RFC3339>] [-store hnsw|qdrant] [-qdrant-url <url>] [-qdrant-collection <name>] [-embed-url <url>] [-embed-model <name>] [-no-embed] [-seed <n>] <stimulus>"

type cliOptions struct {
	specPath         string
	stimulus         string
	persona          string
	personaSet       bool
	previousContext  string
	depth            int
	depthSet         bool
	poolSize         int
	poolSizeSet      bool
	minP             float64
	minPSet          bool
	now              string
	store            string
	qdrantURL        string
	qdrantCollection string
	embedURL         string
	embedModel       string
	noEmbed          bool
	seed             *int64
}

func main() {
	defaultEmbedURL := envOrDefault("BRAINGATE_EMBED_URL", braingate.DefaultEmbedBaseURL)
	defaultEmbedModel := envOrDefault("BRAINGATE_EMBED_MODEL", braingate.DefaultEmbedModel)
	defaultStore := envOrDefault("BRAINGATE_STORE", "hnsw")
	defaultQdrantURL := envOrDefault("BRAINGATE_QDRANT_URL", braingate.DefaultQdrantBaseURL)
	defaultQdrantCollection := envOrDefault("BRAINGATE_QDRANT_COLLECTION", braingate.DefaultQdrantCollection)

	specPath := flag.String("spec", "", "memory specification in JSON")
	personaFlag := flag.String("persona", "", "override persona description passed only to decisions")
	previousContext := flag.String("context", "", "context preceding the stimulus")
	depth := flag.Int("depth", 0, "override temperament descent depth")
	poolSize := flag.Int("pool-size", 0, "override temperament candidate pool size")
	minPFlag := flag.Float64("min-p", 0, "override temperament min_p cutoff ratio in [0, 1]")
	nowFlag := flag.String("now", "", "evaluation timestamp in RFC3339 format (default: current time)")
	store := flag.String("store", defaultStore, "memory store backend: hnsw or qdrant")
	qdrantURL := flag.String("qdrant-url", defaultQdrantURL, "Qdrant REST base URL")
	qdrantCollection := flag.String("qdrant-collection", defaultQdrantCollection, "Qdrant collection name")
	embedURL := flag.String("embed-url", defaultEmbedURL, "OpenAI-compatible embeddings base URL")
	embedModel := flag.String("embed-model", defaultEmbedModel, "embedding model name")
	noEmbed := flag.Bool("no-embed", false, "disable embeddings and rank candidates by affect distance only")
	seed := flag.Int64("seed", 0, "optional PRNG seed for deterministic pool sampling")
	flag.Usage = func() {
		_, _ = fmt.Fprintln(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()

	var personaSet, depthSet, poolSizeSet, minPSet, seedSet bool
	flag.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "persona":
			personaSet = true
		case "depth":
			depthSet = true
		case "pool-size":
			poolSizeSet = true
		case "min-p":
			minPSet = true
		case "seed":
			seedSet = true
		}
	})

	opts := cliOptions{
		specPath:         *specPath,
		stimulus:         strings.Join(flag.Args(), " "),
		persona:          *personaFlag,
		personaSet:       personaSet,
		previousContext:  *previousContext,
		depth:            *depth,
		depthSet:         depthSet,
		poolSize:         *poolSize,
		poolSizeSet:      poolSizeSet,
		minP:             *minPFlag,
		minPSet:          minPSet,
		now:              *nowFlag,
		store:            *store,
		qdrantURL:        *qdrantURL,
		qdrantCollection: *qdrantCollection,
		embedURL:         *embedURL,
		embedModel:       *embedModel,
		noEmbed:          *noEmbed,
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

func envOrDefault(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
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
	storeFactory := func(kind string) (braingate.MemoryStore, error) {
		switch kind {
		case "qdrant":
			return braingate.NewQdrantStore(braingate.QdrantConfig{
				BaseURL:    opts.qdrantURL,
				Collection: opts.qdrantCollection,
				APIKey:     os.Getenv("QDRANT_API_KEY"),
			})
		case "hnsw":
			return braingate.NewHNSWStore(braingate.HNSWConfig{}), nil
		default:
			return nil, fmt.Errorf("unsupported store %q (want hnsw or qdrant)", kind)
		}
	}
	return runWith(ctx, newClientFromEnv, embedFactory, storeFactory, opts, out)
}

func runWith(
	ctx context.Context,
	getClient func() (*decisions.Client, error),
	getEmbedder func() (braingate.Embedder, error),
	getStore func(kind string) (braingate.MemoryStore, error),
	opts cliOptions,
	out io.Writer,
) error {
	if strings.TrimSpace(opts.specPath) == "" || strings.TrimSpace(opts.stimulus) == "" {
		return errors.New(usage)
	}
	if opts.depth < 0 || (opts.depthSet && opts.depth == 0) {
		return errors.New("depth must be >= 1")
	}
	if opts.poolSize < 0 || (opts.poolSizeSet && opts.poolSize == 0) {
		return errors.New("pool-size must be >= 1")
	}
	if opts.minPSet && (math.IsNaN(opts.minP) || math.IsInf(opts.minP, 0) || opts.minP < 0 || opts.minP > 1) {
		return fmt.Errorf("invalid -min-p ratio %v (want [0, 1])", opts.minP)
	}
	var parsedNow time.Time
	if trimmedNow := strings.TrimSpace(opts.now); trimmedNow != "" {
		parsed, err := time.Parse(time.RFC3339, trimmedNow)
		if err != nil {
			return fmt.Errorf("invalid -now timestamp %q (want RFC3339): %w", opts.now, err)
		}
		parsedNow = parsed
	}

	storeKind := strings.ToLower(strings.TrimSpace(opts.store))
	if storeKind == "" {
		storeKind = "hnsw"
	}
	if storeKind != "hnsw" && storeKind != "qdrant" {
		return fmt.Errorf("unsupported store %q (want hnsw or qdrant)", opts.store)
	}

	engine, err := loadSpec(opts.specPath)
	if err != nil {
		return err
	}
	if opts.personaSet {
		engine.Persona = strings.TrimSpace(opts.persona)
	}
	if opts.depth > 0 {
		engine.Temperament.Depth = opts.depth
	}
	if opts.poolSize > 0 {
		engine.Temperament.PoolSize = opts.poolSize
	}
	if opts.minPSet {
		engine.Temperament.MinP = opts.minP
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

	if storeKind == "qdrant" {
		if getStore == nil {
			return errors.New("store factory is required for qdrant")
		}
		store, err := getStore(storeKind)
		if err != nil {
			return err
		}
		memories := engine.Memories
		if embedder != nil {
			memories, err = braingate.EmbedMemories(ctx, embedder, memories)
			if err != nil {
				return err
			}
		}
		if err := store.Upsert(ctx, memories); err != nil {
			return err
		}
		engine.Store = store
		engine.Memories = nil
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
		Now:      parsedNow,
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
