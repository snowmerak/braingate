package incidentdemo

import (
	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

// SampleInput describes a deployment with an observed error spike and no established cause.
const SampleInput = "13시 배포 후 13시 05분부터 오류율이 올랐습니다. 오류 로그에 서비스와 버전이 남아 있습니다. 원인은 아직 모릅니다."

// NewGraph builds a workflow that turns an incident report into a next check.
func NewGraph() *braingate.Neuron {
	nextCheck := &braingate.Neuron{
		ID:           "next_check",
		Instructions: "Does the input say that the error logs record both the affected service and its version?",
		Choices: []braingate.Choice{
			{
				Choice: decisions.Choice{Value: "yes"},
				Text:   "첫 오류 로그의 서비스와 버전을 이번 배포 대상과 대조하세요.",
			},
			{
				Choice: decisions.Choice{Value: "no"},
				Text:   "첫 오류가 발생한 서비스와 버전을 먼저 파악하세요.",
			},
		},
	}
	cause := &braingate.Neuron{
		ID:           "cause",
		Instructions: "Does the evidence establish that the deployment caused the errors? Timing alone is insufficient.",
		Choices: []braingate.Choice{
			{
				Choice: decisions.Choice{Value: "yes"},
				Text:   "이번 배포에서 오류를 일으킨 변경을 확인하세요.",
			},
			{
				Choice: decisions.Choice{Value: "no"},
				Next:   nextCheck,
				Text:   "배포가 원인인지 확인하려면",
			},
		},
	}
	return &braingate.Neuron{
		ID:           "observation",
		Instructions: "Did the error rate rise shortly after deployment?",
		Choices: []braingate.Choice{
			{
				Choice: decisions.Choice{Value: "yes"},
				Next:   cause,
				Text:   "배포 직후 오류율이 올랐습니다.",
			},
			{
				Choice: decisions.Choice{Value: "no"},
				Text:   "오류율이 오르기 시작한 시점을 확인하세요.",
			},
		},
	}
}
