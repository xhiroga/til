package retinacostexplorer

func Strings(strings []string) []*string {
	result := []*string{}
	for i := range strings {
		result = append(result, &strings[i])
	}
	return result
}

func StringsValues(strings []*string) []string {
	result := []string{}
	for _, s := range strings {
		result = append(result, *s)
	}
	return result
}
