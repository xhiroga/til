package stringer

import "testing"

func TestReverse(t *testing.T) {
	t.Run("pineapple", func(t *testing.T) {
		actual := "elppaenip"
		if got := Reverse("pineapple"); got != actual {
			t.Errorf("Expected %s, actual %s", got, actual)
		}
	})
}
