import { StyleSheet } from 'react-native';

export default StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        backgroundColor: '#dcdcdc',
    },
    todos: {
        marginTop: 20,
    },
    cell: {
        margin: 5,
        borderRadius: 5,
        backgroundColor: 'white',
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    checkbox: {
        flex: 5,
        margin: 3,
    },
    deleteButton: {
        flex: 1,
        margin: 3,
        borderRadius: 5,
        backgroundColor: 'red',
        justifyContent: 'center',
    },
    deleteLabel: {
        color: 'white',
        textAlign: 'center',
        textAlignVertical: 'center',
    },
    input: {
        height: 36,
        margin: 2,
        flexDirection: 'row',
        justifyContent: 'center',
    },
    textinput: {
        flex: 1,
        marginRight: 3,
        borderRadius: 5,
        backgroundColor: 'white',
        fontSize: 13,
    },
});
