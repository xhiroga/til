# AUth0 Organization demo

```shell
npm install && npm start
```

## Q&A

- ユーザーが複数のOrganizationに所属するとき、OrganizationごとにRoleを設定できる？
  - 可能。[Add Roles to Organization Members](https://auth0.com/docs/manage-users/organizations/configure-organizations/add-member-roles)
- Organizationのメタデータで、サービスの有効化・無効化を管理できる？（RoleのPermissionをOrganizationのメタデータでFilterできる？）
  - Rule次第で可能と思われる。[OrganizationオブジェクトはContextから取得可能](https://auth0.com/docs/customize/rules/context-object)。
- ユーザーがOrganizationの入力を省略するためのURLを生成できる？
  - [`/authorize` にリダイレクトする際にパラメータを付与することで可能とも読める](https://auth0.com/docs/manage-users/organizations/custom-development#i-want-users-to-log-in-to-a-specified-organization)が、実際に試さないと分からない。

## References ans Inspirations

- [Auth0 Organizationsという素晴らしい機能を今更ながら紹介する](https://zenn.dev/urmot/articles/8c18d8b49d822c#%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB)
- [Auth0 React SDK Quickstarts: Login](https://auth0.com/docs/quickstart/spa/react/01-login?download=true)
- [Auth0 Organizations](https://auth0.com/docs/manage-users/organizations)

## Original Author

[Auth0](https://auth0.com)

## License

This project is licensed under the MIT license. See the [LICENSE](../LICENSE) file for more info.
