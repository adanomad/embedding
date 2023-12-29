import weaviate from "weaviate-ts-client";

export const client = weaviate.client({
  scheme: "http",
  host: "glassbox.ds:8080",
});

// export const getEmbeddingVector = async (fileName: string) => {
//   try {
//     const results = await client.graphql
//     .get()
//     .withClassName("FileEmbedding")
//     .withNearVector({ vector: embedding }) //distance: 0.68
//     .withLimit(limit)
//     .withOffset(offset)
//     .withFields("fileName _additional { distance }")
//     // .withAutocut(1)
//     .do();

//     const result = await client.graphql.get({
//       className: "FileEmbedding",
//       properties: ["embedding"], // Replace with your actual vector property name
//       where: {
//         path: ["fileName"],
//         operator: "Equal",
//         valueString: fileName,
//       },
//     });

//     if (result.data.Get.Image.length > 0) {
//       return result.data.Get.Image[0].embedding;
//     } else {
//       return null; // No matching entry found
//     }
//   } catch (error) {
//     console.error("Error fetching embedding vector:", error);
//     throw error;
//   }
// };
