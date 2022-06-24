
class Notion:
    
    def __init__(self, notion_token):
        self.headers = {
          'Notion-Version': '2021-05-13',
          'Authorization': 'Bearer ' + notion_token
        }
        self.base_url = "https://api.notion.com/v1"
        
    def search_page(self, page_title: str = None):
        
        url = self.base_url + "/search"
        body = {}
        if page_title is not None:
            body["query"] = page_title
        
        response = requests.request("POST", url, headers=self.headers, params=body)
        
        return self.response_or_error(response)
    
    def append_child_blocks(self, parent_id: str, children: []):
        
        url = self.base_url + f"/blocks/{parent_id}/children"
        
        response = requests.request(
        "PATCH",
        url,
        headers=self.headers,
        json={"children": children}
        )
        
        return response
    
    def text_append(self, parent_id: str, text: str):
        
          text_block = {
            "type": "paragraph",
            "paragraph": {"text": [{"type": "text", "text": {"content": text,}}]}
          }
            
        return self.append_child_blocks(parent_id, [text_block])
            
    def update_block(self, block_id: str, content: dict):
        
        url = self.base_url + f"/blocks/{block_id}"
        response = requests.request("PATCH", url, headers=self.headers, json=content)
        
        return self.response_or_error(response)
    
    def text_set(self, block_id: str, new_text: str):
        
        block = self.get_block(block_id)
        type = block["type"]
        block[type]["text"][0]["text"]["content"] = new_text
        
        return self.update_block(block_id, block)
    
    def image_add(self, parent_id: str, image_url: str):
        
		append_children = [
	    {
        "type": "image",
        "image": {
          "type": "external",
          "external": {
            "url": image_url
          }
        }
	    }
		]
		
		return self.append_child_blocks(parent_id, append_children)
    
    def delete_block(self, block_id: str):
        
        url = self.base_url + f"/blocks/{block_id}"
        response = requests.request("DELETE", url, headers=self.headers)
        
        return response
