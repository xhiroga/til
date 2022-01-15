import React, { Component } from 'react';
import { Text, View, StyleSheet, TextInput, Button, Picker } from 'react-native';
import { Constants } from 'expo';
import { Formik } from 'formik';
// You can import from local files
// import AssetExample from './components/AssetExample';

// or any pure javascript modules available in npm
// import { Card } from 'react-native-elements'; // Version can be specified in package.json

export default class Form extends Component {

    render() {
        this.state = { language: '' }
        return (
            <View style={styles.container}>
                <Text style={styles.paragraph}>
                    Formik x React Native
        </Text>
                <Formik
                    initialValues={{ firstName: '' }}
                    onSubmit={values => console.log(values)}>
                    {({ handleChange, handleSubmit, values }) => (
                        <View>
                            <TextInput
                                style={{
                                    height: 40,
                                    borderColor: 'gray',
                                    borderWidth: 1,
                                    width: 300,
                                    padding: 8,
                                    fontSize: 18
                                }}
                                onChangeText={handleChange('firstName')}
                                value={values.firstName}
                            />
                            <Button onPress={handleSubmit} title="submit" color="#841584" />
                        </View>
                    )}
                </Formik>

                <Picker
                    selectedValue={this.state.language}
                    style={{ height: 50, width: 100 }}
                    onValueChange={(itemValue, itemIndex) => this.setState({ language: itemValue })}>
                    <Picker.Item label="Java" value="java" />
                    <Picker.Item label="JavaScript" value="js" />
                </Picker>
            </View>
        );
    }
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: 'center',
        // justifyContent: 'center',
        paddingTop: Constants.statusBarHeight + 100,
        backgroundColor: '#ecf0f1',
    },
    paragraph: {
        margin: 24,
        fontSize: 18,
        fontWeight: 'bold',
        textAlign: 'center',
        color: '#34495e',
    },
});
