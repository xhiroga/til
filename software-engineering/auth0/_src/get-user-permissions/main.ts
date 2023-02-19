import { ManagementClient } from 'auth0'
import * as dotenv from 'dotenv'

dotenv.config()

const run = async () => {
    // @note -- 'read:users' is needed.
    const management = new ManagementClient({
        domain: process.env.DOMAIN!!,
        clientId: process.env.CLIENT_ID,
        clientSecret: process.env.CLIENT_SECRET,
        scope: 'read:users'
    })
    const permissions = await management.getUserPermissions({ id: process.env.USER_ID!! })
    console.log(permissions, permissions.map(p => p.permission_name))
}
run()
