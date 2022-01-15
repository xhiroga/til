import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "cdk-python-dotvscode" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('cdk-python-dotvscode.dotvscode-dir-and-setting-json', async () => {
		const getWorkspaceDirPath = () => {
			if (vscode.workspace.workspaceFolders !== undefined){
				return vscode.workspace.workspaceFolders[0].uri
			}
			return undefined
		}
		
		const isCdkProject = async (workspaceDirPath: vscode.Uri) => {
			return await vscode.workspace.fs.stat(vscode.Uri.file(`${workspaceDirPath.toString().split(':')[1]}/cdk.out`)).then( stat =>{
				return stat.type === vscode.FileType.Directory
			})
		}

		const setConfig  = async (workspaceDirPath: vscode.Uri) => {
			const config = vscode.workspace.getConfiguration('python', workspaceDirPath)
			// to specify where libraries such as aws-cdk were installed.
			config.update("pythonPath", "${workspaceFolder}/.env/bin/python", false)
			config.update("envFile", "${workspaceFolder}/.env", false)
		}

		const workspaceDirPath  = getWorkspaceDirPath()
		if (workspaceDirPath !== undefined && isCdkProject(workspaceDirPath)){
			setConfig(workspaceDirPath)
			vscode.window.showInformationMessage(
				".vscode and setting.json was created"
			)
		}else {
			// TODO
		}
	});

	context.subscriptions.push(disposable);
}

// this method is called when your extension is deactivated
export function deactivate() {}
