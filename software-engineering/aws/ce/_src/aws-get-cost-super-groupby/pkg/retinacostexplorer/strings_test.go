package retinacostexplorer

import (
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
)

func TestStrings(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		out  []*string
	}{
		{
			name: "empty",
			in:   []string{},
			out:  []*string{},
		},
		{
			name: "success",
			in:   []string{"a", "b"},
			out:  []*string{aws.String("a"), aws.String("b")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Strings(tt.in); !reflect.DeepEqual(StringsValues(got), StringsValues(tt.out)) {
				t.Errorf("Strings() = %v, want %v", StringsValues(got), StringsValues(tt.out))
			}
		})
	}
}

func TestStringsValues(t *testing.T) {
	tests := []struct {
		name string
		in   []*string
		out  []string
	}{
		{
			name: "empty",
			in:   []*string{},
			out:  []string{},
		},
		{
			name: "success",
			in:   []*string{aws.String("a"), aws.String("b")},
			out:  []string{"a", "b"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := StringsValues(tt.in); !reflect.DeepEqual(got, tt.out) {
				t.Errorf("StringsValues() = %v, want %v", got, tt.out)
			}
		})
	}
}
