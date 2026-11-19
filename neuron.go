package braingate

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/snowmerak/decisions"
)

// ErrCycle indicates that the selected route revisited a neuron in the same run.
var ErrCycle = errors.New("braingate: cycle in neuron route")

// Choice is a Decisions option and its next neuron. A nil Next ends the run.
type Choice struct {
	decisions.Choice
	Next *Neuron

	// Text is an optional phrase or sentence contributed by this choice.
	Text string
}

// Neuron asks one Decisions choice question. Use a distinct ID for each neuron
// so it can be identified in the selection history. Configure the graph before
// calling Run and keep it fixed while runs are in progress.
type Neuron struct {
	ID           string
	Instructions string
	Choices      []Choice
}

// Selection records one neuron's answer in the order it was produced.
type Selection struct {
	NeuronID      string             `json:"neuron_id"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Text          string             `json:"text,omitempty"`
}

// ComposeText joins the selected text in execution order with single spaces.
// Choices supply their own punctuation. Blank text contributes nothing.
func ComposeText(history []Selection) string {
	parts := make([]string, 0, len(history))
	for _, selection := range history {
		if part := strings.TrimSpace(selection.Text); part != "" {
			parts = append(parts, part)
		}
	}

	return strings.Join(parts, " ")
}

type decisionState struct {
	Input   any         `json:"input"`
	History []Selection `json:"history"`
}

// Run starts at n and follows the selected choice at each neuron. Every
// question receives the initial input state and all prior selections, including
// their text, under the JSON keys input and history, along with the original
// assets. Run returns the selections completed before any error.
func (n *Neuron) Run(ctx context.Context, client *decisions.Client, input decisions.Input) ([]Selection, error) {
	if n == nil {
		return nil, errors.New("braingate: starting neuron is required")
	}

	state := decisionState{Input: input.State, History: []Selection{}}
	visited := make(map[*Neuron]struct{})
	for neuron := n; neuron != nil; {
		if strings.TrimSpace(neuron.ID) == "" {
			return state.History, errors.New("braingate: neuron ID is required")
		}
		if _, seen := visited[neuron]; seen {
			return state.History, fmt.Errorf("%w: %q", ErrCycle, neuron.ID)
		}
		visited[neuron] = struct{}{}

		options := make([]decisions.Choice, 0, len(neuron.Choices))
		for _, choice := range neuron.Choices {
			options = append(options, choice.Choice)
		}
		input.State = state
		result, err := client.Choices(ctx, input, decisions.ChoicesQuestion{
			Instructions: neuron.Instructions,
			Choices:      options,
		})
		if err != nil {
			return state.History, fmt.Errorf("braingate: neuron %q: %w", neuron.ID, err)
		}

		answer := result.Answer
		selected := neuron.choice(answer.Choice)
		state.History = append(state.History, Selection{
			NeuronID:      neuron.ID,
			Choice:        answer.Choice,
			Confidence:    answer.Confidence,
			Probabilities: answer.Probabilities,
			Text:          selected.Text,
		})
		neuron = selected.Next
	}

	return state.History, nil
}

func (n *Neuron) choice(value string) Choice {
	for _, choice := range n.Choices {
		if choice.Value == value {
			return choice
		}
	}
	return Choice{}
}
