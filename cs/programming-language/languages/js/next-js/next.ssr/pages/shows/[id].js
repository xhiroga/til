import fetch from "isomorphic-unfetch";

const Post = (props) => (
  <div>
    <h1>{props.show.name}</h1>
    <p>{props.show.summary.replace(/<[/]?[pb]>/g, "")}</p>
    {props.show.image ? <img src={props.show.image.medium} /> : null}
  </div>
);

Post.getInitialProps = async function (context) {
  const { id } = context.query;
  const res = await fetch(`https://api.tvmaze.com/shows/${id}`);
  const show = await res.json();
  console.log(`Fetched show: ${show.name}`);
  return { show };
};

export default Post;
